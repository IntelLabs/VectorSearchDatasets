import argparse
import numpy as np
import text_dataset_generate as tdg


def main(embed_model, number_vec_shards, embed_size, batch_size, tmp_output_dir, output_prefix, final_output_dir,
         data_dir, dataset_name, chunk_size=None, json_name=None, hf_path=None,
         start_vector=0, end_vector=None, data_key='contents', multiproc=True, combine_only=False):
    data = tdg.load_dataset_hf_or_jsonl(dataset_name, data_dir, json_name, hf_path)
    n_samples = len(data)
    if combine_only:
        print("Skipping embedding, assuming files exist already...")
    else:
        if number_vec_shards > 1:
            assert end_vector is None, "End vector should be none if you're sharding the data"
            shard_edges = np.arange(0, n_samples, n_samples // number_vec_shards)
            shard_start_indices = shard_edges[:-1]
            shard_end_indices = shard_edges[1:]
            do_shard = shard_end_indices > start_vector
            shard_start_indices = shard_start_indices[do_shard]
            for start_index in shard_start_indices:
                start_vec = start_index if start_index >= start_vector else start_vector
                tdg.generate_text_embeddings(data, embed_model, number_vec_shards, embed_size, batch_size, chunk_size,
                                             tmp_output_dir, output_prefix, start_vec, end_vector, data_key, multiproc)
        else:
            tdg.generate_text_embeddings(data, embed_model, number_vec_shards, embed_size, batch_size, chunk_size,
                                         tmp_output_dir, output_prefix, start_vector, end_vector, data_key, multiproc)
    del data

    # Combine & convert the numpy files to fvecs format
    mmap_shape = (n_samples, embed_size)
    output_file_prefix = f"{tmp_output_dir}/{output_prefix}"
    mmap_path = tdg.combine_mmaps(output_file_prefix, mmap_shape)  # if there's only 1 shard this will do nothing
    tdg.convert_mmap_fvecs(mmap_path, mmap_shape, final_output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # embedding model arguments
    parser.add_argument("-m", "--embed_model", type=str, required=True,
                        help="Model ID for SentenceTransformers or HuggingFace")
    parser.add_argument("-e", "--embed_size", type=int, required=True,
                        help="Size of the embedding dimension")
    # dataset arguments
    parser.add_argument("-dn", "--dataset_name", type=str,
                        help="Dataset name, e.g. `kilt_wikipedia`, for saving out the pyarrow dataset files. If you would like "
                             "the dataset to be downloaded from HuggingFace, this should be the `name` argument to "
                             "datasets.load_dataset().")
    parser.add_argument("-dd", "--data_dir", type=str, required=True,
                        help="HuggingFace dataset cache folder")
    parser.add_argument("-dk", "--data_key", type=str, default="contents",
                        help="Name of the dataset field that contains the text")
    parser.add_argument("-jn", "--json_name", type=str, default=None,
                        help="Full path to dataset jsonl file. If None, will HuggingFace download instead.")
    parser.add_argument("-hp", "--hf_path", type=str, default=None,
                        help="HuggingFace path to the dataset, e.g. `jenhsia/ragged`. If None, assumes JSON loading.")
    # data processing arguments
    parser.add_argument("-nv", "--number_vec_shards", type=int, default=1,
                        help="The number of overall shards of the output")
    parser.add_argument("-b", "--batch_size", type=int, default=512,
                        help="Batch size for running inference on embedding model")
    parser.add_argument("-sv", "--start_vector", type=int, default=0,
                        help="Vector ID to start from. Allows you to restart if needed.")
    parser.add_argument("-ev", "--end_vector", type=int, default=None,
                        help="Vector ID to end with. Allows you to fill in data if needed.")
    parser.add_argument("-mp", "--multiproc", action="store_true",
                        help="Enables multi-device inference")
    parser.add_argument("-c", "--chunk_size", type=int, default=None,
                        help="For multiprocessing, a chunk is the number of sentences to send for processing on a "
                             "given device. It's required that chunk_size >= batch_size.")
    # output arguments
    parser.add_argument("-ot", "--tmp_output_dir", type=str, default="/var/tmp/svs",
                        help="Intermediate output directory of memory mapped files.")
    parser.add_argument("-op", "--output_prefix", type=str, required=True,
                        help="Output filename prefix")
    parser.add_argument("-of", "--final_output_dir", type=str, required=True,
                        help="Final output directory of the .fvecs embedding file.")
    parser.add_argument("--combine_only", action="store_true",
                        help="Skip embedding and only do the combination / fvec conversion")
    args = parser.parse_args()
    print(args)

    if args.hf_path is None and args.json_name is None:
        raise ValueError("You must specify where to find the dataset, either through `--hf_path` or `--json_name`.")
    print(args)

    arg_dict = vars(args)

    main(**arg_dict)

