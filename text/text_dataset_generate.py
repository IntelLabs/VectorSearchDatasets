"""
Can be used to generate vector embeddings of a HuggingFace-compatible dataset using SentenceTransformers. Assumes the
use of a text dataset. Has features to parallelize encoding across data samples.

1) Loads the dataset from a JSONL file with the HuggingFace datasets library.
    - The data in the JSONL file should have a field that corresponds to the already chunked text, where each line in
    the JSONL corresponds to a single chunk / single data sample. The name of this data field should be specified here
    with the input argument --data_key.
    - Saves the HuggingFace dataset to disk using pyarrow shards. If this dataset has already been created from a JSONL
    previously, it will simply load it from these shards instead of re-constructing it from the JSONL.
2) Using a model from SentenceTransformers, creates embeddings for each sample.
    - Writes to a memory-mapped numpy file.
    - Can start and stop embeddings with --start_vector and --end_vector indices. This allows you to resume a previously
      terminated job.
    - To handle very large datasets, the dataset processing can be broken into multiple shards. You specify the number
      of shards, and the script will nearly evenly split the dataset into that number of shards. You can specify the
      index of the vector to start and end with for a call to this script. The script will automatically determine
      which shard to write into and resume dataset creation from there. This can enable inter-node data parallelism
      (e.g. one shard per node).
    - To handle very large datasets, you can enable intra-node data parallelism with --multiproc. This allows you to
      recruit each GPU to encode a different portion of the dataset.
"""
import datasets as ds
import gc
import glob
import math
import numpy as np
import numpy.typing as npt
import os
import sentence_transformers as st
import tqdm
import torch.utils.data
from typing import List, Dict, Optional


class MMAPSentenceTransformer(st.SentenceTransformer):
    """
    Subclass to write directly to a memory mapped file with multiprocessing. Overwrites encode_multi_process()
    function to do this. Still relies on the original class' implementation of _encode_multi_process_worker,
    as well as the functions to start and stop the multiprocessing pool.
    
    This modification was written for sentence-transformers==3.0.0 from conda-forge. An older version did not use the
    `precision` argument in the queue, and future versions may require other changes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode_multi_process(
        self,
        mmap_fname: str,
        sentences: List[str],
        pool: Dict[str, object],
        emb_size: int = None,
        batch_size: int = 32,
        chunk_size: int = None,
        start_chunk: int = 0,
        end_chunk: int = None,
        normalize_embeddings: bool = False,
        prompt_name: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        """
        This method allows to run encode() on multiple GPUs. The sentences are chunked into smaller packages
        and sent to individual processes, which encode these on the different GPUs. This method is only suitable
        for encoding large sets of sentences

        :param prompt_name: The name of the prompt to use for encoding. Must be a key in the `prompts` dictionary,
            which is either set in the constructor or loaded from the model configuration. For example if
            `prompt_name` is ``"query"`` and the `prompts` is ``{"query": "query: {}", ...}``, then the sentence "What
            is the capital of France?" will be encoded as "query: What is the capital of France?". If `prompt` is
            also set, this argument is ignored.
        :param prompt: The prompt to use for encoding. For example, if the prompt is ``"query: {}"``, then the
            sentence "What is the capital of France?" will be encoded as "query: What is the capital of France?".
            If `prompt` is set, `prompt_name` is ignored.
        :return: 2d numpy array with shape [num_inputs, output_dimension]
        """
        precision = "float32"
        if os.path.exists(mmap_fname):
            print(f"Opening existing memory mapped file to write into: {mmap_fname} {len(sentences), emb_size}")
            mmap = np.memmap(mmap_fname, dtype='float32', mode="r+",
                             shape=(len(sentences), emb_size))
        else:
            print(f"Creating new memory mapped file to write into: {mmap_fname} {len(sentences), emb_size}")
            mmap = np.memmap(mmap_fname, dtype='float32', mode="w+",
                             shape=(len(sentences), emb_size))

        if chunk_size is None:
            chunk_size = min(math.ceil(len(sentences) / len(pool["processes"]) / 10), 5000)
        if end_chunk is None:
            end_chunk = (len(sentences)//chunk_size) + 1

        print(f"Chunk data into {math.ceil(len(sentences) / chunk_size)} packages of size {chunk_size}")

        input_queue = pool["input"]
        last_chunk_id = 0
        chunk = []

        for sentence in sentences:
            chunk.append(sentence)
            if len(chunk) >= chunk_size:
                if last_chunk_id < start_chunk:
                    chunk = []
                    last_chunk_id += 1
                    continue
                elif last_chunk_id == start_chunk:
                    print(f"Sending sentences starting from chunk {last_chunk_id}, vector index {last_chunk_id*chunk_size}")
                if last_chunk_id >= end_chunk:
                    print(f"Stopping sentences at chunk {last_chunk_id}")
                    chunk = []
                    break
                input_queue.put([last_chunk_id, batch_size, chunk, prompt_name, prompt, precision, normalize_embeddings])
                last_chunk_id += 1
                chunk = []

        if len(chunk) > 0:
            input_queue.put([last_chunk_id, batch_size, chunk, prompt_name, prompt, precision, normalize_embeddings])
            last_chunk_id += 1

        output_queue = pool["output"]
        for _ in tqdm.tqdm(range(start_chunk, last_chunk_id)):
            chunki, embs = output_queue.get()
            starti = chunki*chunk_size
            endi = starti + chunk_size
            mmap[starti:endi, :] = embs
            mmap.flush()
        mmap._mmap.close()
        del mmap
        gc.collect()


def load_dataset_hf_or_jsonl(dataset_name, data_dir, json_name=None, hf_path=None):
    """
    Loads a text dataset, either from a JSONL file or a HuggingFace data repository.
    Assumes the data has already been preprocessed. Each data sample should correspond to a single chunk of text.
    This function implements its own disk caching. This implementation (1) allows easy movement of the data to a new
    machine and (2) ensures that caching is only performed once, and is not repeated if the HuggingFace fingerprint
    changes because of changes to the surrounding code.

    :param dataset_name: Name of the dataset to use in dataset disk caching. If loading from HuggingFace Hub,
                         this should also be the `name` argument to `datasets.load_dataset()`.
    :param data_dir: Directory for the dataset arrow files, e.g. the HuggingFace dataset cache folder.
    :param json_name: default None. Full path to the JSONL file. If None, `hf_path` must be specified.
    :param hf_path: default None. Path to the HuggingFace dataset repository. If None, `json_name` must be specified.
    :return: data, an instance of a datasets.Dataset class.
    """
    # Load dataset.
    shard_proc_path = f"{data_dir}/processed_{dataset_name}"
    if os.path.exists(shard_proc_path):
        print(f"Found previously loaded data, loading from {shard_proc_path}")
        data = ds.load_from_disk(shard_proc_path)
    else:
        print(f"Did not find dataset in {shard_proc_path}...")
        # Preprocess: make each paragraph a separate sample, keep metadata
        print(f"Loading dataset {dataset_name}...")
        if json_name:
            data = ds.load_dataset("json", data_files=f"{json_name}")
        elif hf_path:
            data = ds.load_dataset(hf_path, dataset_name, cache_dir=data_dir)
            print("WARNING: we are assuming you want the `train` split.")
        else:
            raise ValueError("Either json_name or hf_path must be specified!")
        data = data['train']
        # Save the pre-processed version in shards
        data.save_to_disk(shard_proc_path, max_shard_size="500MB")
        print("Saved preprocessed data!")
    return data


def generate_text_embeddings(data, embed_model, number_vec_shards, embed_size, batch_size, chunk_size,
                             tmp_output_dir, output_prefix, start_vector=0, end_vector=None,
                             data_key='text', multiproc=True):
    # Load embedding model
    print(f"Loading embedding model {embed_model}")
    embed_model = MMAPSentenceTransformer(embed_model, trust_remote_code=True)

    # Write to memory mapped numpy file
    os.makedirs(tmp_output_dir, exist_ok=True)
    nsentences = len(data)
    print(f"Found a total of {nsentences} samples in the dataset.")
    if number_vec_shards == 1:
        output_fn = f"{tmp_output_dir}/{output_prefix}_tmp.mmap"
        sentence_data = data[data_key]
        ns = nsentences
        start_vector = start_vector if start_vector else 0
        end_vector = ns if end_vector is None else end_vector
    elif number_vec_shards > 1:
        # Shard the data and write several output files to combine at end
        # Get sample indices for the relevant shard
        shard_edges = np.arange(0, nsentences, nsentences // number_vec_shards)
        shard_edges[-1] = nsentences  # encompass the last item in the bin
        # Determine which shard we're processing
        shard_id = np.digitize(start_vector, shard_edges) - 1
        left_edge, right_edge = shard_edges[shard_id:shard_id+2]
        sentence_data = data[data_key][left_edge:right_edge]
        # Set start & end vectors relative to the shard
        start_vector = start_vector - left_edge
        ns = right_edge - left_edge   # shard size
        end_vector = ns
        output_fn = f"{tmp_output_dir}/{output_prefix}-{shard_id}_tmp.mmap"
    else:
        raise ValueError("number_vec_shards should be >= 1")
    del data

    if multiproc:
        assert torch.cuda.is_available(), "GPUs for multiprocessing not detected."
        print(f"Starting multiprocessing pool...")
        pool = embed_model.start_multi_process_pool()
        print(f"Embedding {ns} vectors...")
        try:
            start_chunk = start_vector // chunk_size
            end_chunk = (end_vector // chunk_size) + 1
            embed_model.encode_multi_process(mmap_fname=output_fn, pool=pool,
                                             emb_size=embed_size,
                                             sentences=sentence_data,
                                             batch_size=batch_size,
                                             chunk_size=chunk_size,
                                             start_chunk=start_chunk,
                                             end_chunk=end_chunk)
        except Exception as e:
            print("Error, closing pool")
            embed_model.stop_multi_process_pool(pool)
            raise e
        embed_model.stop_multi_process_pool(pool)
        del embed_model
        gc.collect()
        print(f"Finished multiprocessing.")
    else:
        if os.path.exists(output_fn):
            print(f"Opening existing memory mapped file to write into: {output_fn}")
            fout = np.memmap(output_fn, dtype=np.float32, mode="r+", shape=(ns, embed_size))
        else:
            fout = np.memmap(output_fn, dtype=np.float32, mode="w+", shape=(ns, embed_size))
        start_chunk = start_vector // batch_size
        end_chunk = (end_vector // batch_size) + 1
        dl = torch.utils.data.DataLoader(sentence_data, batch_size=batch_size, shuffle=False)
        print(f"Embedding {ns} vectors...")
        for bi, batch in enumerate(tqdm.tqdm(dl)):
            if bi < start_chunk:
                continue
            elif bi == start_chunk:
                print(f"Writing embeddings starting from batch {bi}")
            if bi >= end_chunk:
                print(f"Stopping embedding writing at batch {bi}")
                fout.flush()  # Not sure if necessary to flush
                break
            i = bi*batch_size
            j = min(i + batch_size, end_vector)
            fout[i:j, :] = embed_model.encode(batch, batch_size=batch_size)
        fout._mmap.close()
        del fout
        gc.collect()


def fvecs_write_from_mmap(fname: str, m: npt.ArrayLike):
    n, d = m.shape
    m1 = np.memmap(fname, dtype='int32', mode='w+', shape=(n, d + 1))
    m1[:, 0] = d
    m1[:, 1:] = m.view('int32')


def convert_mmap_fvecs(input_fn, mmap_shape, output_fn):
    """
    Given an input file path directing to a memory mapped numpy file, load it and save in
    SVS-compatible fvecs format. The memory mapped file needs to have a shape input so
    it can be loaded by numpy.
    """
    if output_fn[-6:] != '.fvecs':
        # Output file name should end in .fvecs
        if os.path.isdir(output_fn):
            base_file = os.path.basename(input_fn).split('.')[0]
            output_fn = f"{output_fn}/{base_file}.fvecs"
        else:
            path = os.path.dirname(output_fn)
            base = os.path.basename(output_fn).split('.')[0]
            output_fn = f"{path}/{base}.fvecs"
    print(f"Reading data from {input_fn} to write to {output_fn}")
    fdata = np.memmap(input_fn, dtype='float32', mode="r", shape=mmap_shape)
    print(f"Converting to fvecs...")
    fvecs_write_from_mmap(output_fn, fdata)
    print(f"Finished with {output_fn}!")


def combine_mmaps(data_file_prefix, final_mmap_shape):
    output_file_path = data_file_prefix + '_combined.mmap'
    files_to_combine = glob.glob(f'{data_file_prefix}*[0-9]*_tmp.mmap')
    n_files = len(files_to_combine)
    files_to_combine = sorted(files_to_combine, key=lambda i: int(os.path.basename(i).split('-')[-1].split('_')[0])) if n_files > 1 else files_to_combine
    if os.path.exists(output_file_path):
        outf = np.memmap(output_file_path, dtype='float32', mode="r+", shape=final_mmap_shape)
        rand_inds = np.concatenate((np.array([0, -1, -2, -10]),
                                    np.random.randint(0, final_mmap_shape[0] - 2, 10)))
        if np.any(np.sum(outf[rand_inds, :], axis=1) == 0):
            print(f"Found combined mmap file, but some part of it is empty so overwriting!")
            print(f"Combining memmap data across {n_files} files:\n{files_to_combine}")
            outf = np.memmap(output_file_path, dtype='float32', mode="w+", shape=final_mmap_shape)
        else:
            print(f"Found combined mmap file! Reusing...")
            return output_file_path
    elif n_files > 1:
        print(f"Combining memmap data across {n_files} files:\n{files_to_combine}")
        outf = np.memmap(output_file_path, dtype='float32', mode="w+", shape=final_mmap_shape)
    elif n_files == 0:
        unsharded_file = glob.glob(f'{data_file_prefix}*.mmap')
        if len(unsharded_file) == 1:
            return unsharded_file[0]
        else:
            raise ValueError(f"Could not find any files following {data_file_prefix}*.mmap!")
    elif n_files == 1:
        print(f"Found one mmap file, assuming no combination is necessary...")
        return files_to_combine[0]
    nsamples = final_mmap_shape[0]
    shard_edges = np.arange(0, nsamples, nsamples // n_files)
    shard_edges[-1] = nsamples
    file_dim0 = np.diff(shard_edges)
    print(f"Shards at {shard_edges}, files contain {file_dim0} samples")
    starti = 0
    for d, fn in zip(file_dim0, files_to_combine):
        try:
            indata = np.memmap(fn, dtype='float32', mode="r", shape=(d, final_mmap_shape[1]))
            print(f"Copying over data from {fn}")
        except Exception as e:
            print(e)
            import pdb; pdb.set_trace()
        outf[starti:starti+d, :] = indata
        del indata
        starti += d
    del outf
    print(f"Done combining!")
    return output_file_path
