import numpy as np
import os, shutil
import gzip
import random
from datasets import load_dataset
from typing import Optional
import nltk
import numpy.typing as npt
import rocketqa
import toml


def generate_rqa_embeddings(
        dual_encoder: rocketqa.encoder.dual_encoder.DualEncoder,
        init_file: int,
        number_of_files: int,
        c4_dataset_dir: str,
        dataset_dir: str,
        fname_prefix_out: str,
        cache_folder_hugg: str,
        num_embd: Optional[int] = None,
        generate_queries: Optional[bool] = False,
        questionRequired: Optional[bool] = True):
    """Generate rocketQA embeddings from text snippets of the C4 dataset and save them in .fvecs format.

    Keyword arguments:
    dual_encoder -- RocketQA dual encoder, e.g., load using rocketqa.load_model(model="v1_marco_de")
    init_file -- Use the init_file and number_of_files parameters to determine the range of files to extract the text
                snippets from. For example, when generating base vectors, init_file=5 and number_of_files=10 will
                process files c4-train.00005-of-01024.json.gz to c4-train.00014-of-01024.json.gz from the C4/en dataset.
    number_of_files -- See description for init_file.
    c4_dataset_dir -- Path to the folder where the C4 dataset is located.
    dataset_dir -- Path where the new dataset files will be saved.
    fname_prefix_out -- Prefix used for the names of the .fvecs files where the embeddings are saved.
    cache_folder_hugg -- Path to the huggingface cache folder.
    num_embd -- Optional; Number of vector embeddings to generate. If not specified, the maximum number of embeddings
                available in the input files are generated. If the input files do not contain enough text snippets to
                generate the requested number of embeddings a warning will be printed and the .fvecs file will be saved
                with the generated embeddings. Add more files if more embeddings are needed.
    generate_queries -- Optional; Set to True to generate queries with the query encoder
                        (dual_encoder.encode_query()). Otherwise, the base vectors are generated with the
                        paragraph encoder (dual_encoder.encode_para()).
    questionRequired -- Optional; If set to True, sentences with question marks are prioritized during query generation.
    """

    end_file = init_file + number_of_files

    cache_folder = f'{cache_folder_hugg}/files{init_file}_{end_file}'
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    if generate_queries:
        total_files = 8
        fname_prefix = 'c4-validation.'
    else:
        total_files = 1024
        fname_prefix = 'c4-train.'

    fname_fvecs_temp = f'{dataset_dir}/fvecs/{fname_prefix_out}_files{init_file}_{end_file}_temp.fvecs'
    fname_fvecs = f'{dataset_dir}/fvecs/{fname_prefix_out}_files{init_file}_{end_file}.fvecs'

    if not os.path.exists(f'{dataset_dir}/fvecs'):
        os.makedirs(f'{dataset_dir}/fvecs')

    file_id = init_file
    curr_total_emb = 0
    first_file = True

    while file_id < end_file:

        if num_embd is not None and curr_total_emb >= num_embd:
            print(f'Number of requested embeddings reached {num_embd}.')
            break

        dataset_fname = f'{c4_dataset_dir}{fname_prefix}{file_id:05d}-of-{total_files:05d}'

        print('Processing file' + dataset_fname)
        # Unzip file
        with gzip.open(dataset_fname + '.json.gz', 'rb') as f_in:
            with open(dataset_fname + '.json', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Load json from decompressed file
        dataset = load_dataset('json', data_files=dataset_fname + '.json', cache_dir=cache_folder)

        # Remove unnecesary json to save storage
        os.remove(dataset_fname + '.json')

        if generate_queries:
            queries, text_ids = extract_queries(dataset['train']['text'], questionRequired=questionRequired,
                                                nRequired=num_embd)
            if num_embd is not None:
                queries = queries[:num_embd]
            print('Encoding', len(queries), 'queries.')
            embeddings_batch = np.array(list(dual_encoder.encode_query(query=queries)))
            print(f'Generated query embeddings of size: {embeddings_batch.shape}')

        else:
            contexts = dataset['train']['text']
            print('Encoding', len(contexts), 'contexts.')
            embeddings_batch = np.array(list(dual_encoder.encode_para(para=contexts)))
            print(f'Generated context embeddings of size: {embeddings_batch.shape}')

        if first_file:
            embeddings = embeddings_batch
            first_file = False
        else:
            embeddings = np.concatenate([embeddings, embeddings_batch])

        curr_total_emb = embeddings.shape[0]
        print(curr_total_emb, 'embeddings generated so far,', embeddings_batch.shape[0], 'from file', dataset_fname)

        print('Cleaning huggingface cache...')
        cache_folder += '/json/'
        for filename in os.listdir(cache_folder):
            file_path = os.path.join(cache_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path, ignore_errors=True)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

        file_id += 1

        if file_id < end_file:
            print(f'Saving intermediate fvecs file with {curr_total_emb} embeddings...')
            write_fvecs(fname_fvecs_temp, embeddings)

    if num_embd is not None:
        print(f'Saving {embeddings[:num_embd].shape[0]} embeddings to .fvecs format.')
        write_fvecs(fname_fvecs, embeddings[:num_embd])

        if embeddings[:num_embd].shape[0] < num_embd:
            print(f'WARNING: the input files did not contain enough text snippets to generate {num_embd} embeddings. '
                  f'Only {curr_total_emb} embeddings were generated. Add more input files to achieve the required number of'
                  f' embeddings!')
    else:
        # Save embeddings to fvecs format
        print(f'Saving {embeddings.shape[0]} embeddings to .fvecs format.')
        write_fvecs(fname_fvecs, embeddings)

    try:
        print('Deleting temporal .fvecs files')
        os.remove(fname_fvecs_temp)
    except OSError:
        pass


def merge_embedding_files(init_file: int,
                          number_of_files: int,
                          final_file: int,
                          dataset_dir: str,
                          fname_prefix_out: str
                          ) -> npt.ArrayLike:
    f""" Merge the embeddings from .fvecs files generated using function generate_rqa_embeddings. The files 
         to be merged are named: 
         
         [dataset_dir]/fvecs/[fname_prefix_out]_files[init_file]_[end_file].fvecs         

    Keyword arguments:   
    init_file -- Id of the initial C4 text file processed.          
    number_of_files -- Same number_of_files used by generate_rqa_embeddings.    
    final_file -- Id of the first file of the last batch that is to be merged (e.g., if processing files 0 to 32 in 
                  batches of 4, final_file = 28)
    dataset_dir -- Path to the location of .fvecs files generated using generate_rqa_embeddings. The files must be
                located at dataset_dir/fvecs.
    fname_prefix_out -- Same fname_prefix_out used by generate_rqa_embeddings.
    """

    # Loading first file
    end_file = init_file + number_of_files
    fname_fvecs = f'{dataset_dir}/fvecs/{fname_prefix_out}_files{init_file}_{end_file}.fvecs'
    X = read_fvecs(fname_fvecs)
    print(f'Loading {X.shape[0]} embeddings from file: {fname_fvecs}')

    # Loading following files
    for i, init_file in enumerate(range(init_file + number_of_files, final_file + number_of_files, number_of_files)):
        end_file = init_file + number_of_files
        fname_fvecs = f'{dataset_dir}/fvecs/{fname_prefix_out}_files{init_file}_{end_file}.fvecs'
        Xaux = read_fvecs(fname_fvecs)

        X = np.concatenate((X, Xaux))
        print(f'Loading {Xaux.shape[0]} embeddings from file: {fname_fvecs}, total embeddings: {X.shape[0]}')

    return X


def split_base_query_learn_sets(X: npt.ArrayLike,
                                init_file: int,
                                final_file: int,
                                fname_prefix_out: str,
                                dataset_dir_out: str,
                                save_ids_shuffling: Optional[bool] = True,
                                fname_rqa_10m_multiplicity: Optional[int] = None):
    f""" Split the generated embeddings into base, query and learning sets (ID and OOD) and save the corresponding 
         files. Two sets of base vectors are created, one containing 10M and the other containing 1M vectors. 
         The query and learning sets contain 10k vectors each.

    Keyword arguments:        
    X -- Embeddings to be split.
    init_file -- Id of the initial C4 text file processed. 
    final_file -- Id of the first file of the last batch that is to be merged (e.g., if processing files 0 to 32 in 
                  batches of 4, final_file = 28)
    fname_prefix_out -- Same fname_prefix_out used by generate_rqa_embeddings.
    dataset_dir -- Directory where the output dataset files will be saved.
    save_ids_shuffling -- Optional; Set to True to save the id shuffling used to split the dataset.
    fname_rqa_10m_multiplicity -- Optional; .toml file containing a list of duplicate elements to be removed from the 
                                  10M version of the dataset.  
    """

    np.random.seed(0)
    random.seed(0)

    nb = 10000000  # Base number of vectors
    nq = 10000  # Number of queries
    nl = 10000  # Number of learning vectors

    if not os.path.exists(f'{dataset_dir_out}/fvecs'):
        os.makedirs(f'{dataset_dir_out}/fvecs')
    if not os.path.exists(f'{dataset_dir_out}/npz'):
        os.makedirs(f'{dataset_dir_out}/npz')

    # Shuffle ids and keep only the requested n embeddings. The shuffling is done in order to split the vectors into
    # base, query and learning IID sets.
    ids = np.arange(X.shape[0])
    np.random.shuffle(ids)
    if save_ids_shuffling:
        # Save ids shuffling for reproducibility
        np.savez(f'{dataset_dir_out}/npz/{fname_prefix_out}_ids_mapping_files{init_file}_{final_file}.npz', ids=ids)

    # Use the first 10k as a in-distribution query set
    write_fvecs(f'{dataset_dir_out}/rqa_query_{int(nq / 1000)}k.fvecs', X[ids[:nq]])
    print(f'Saving {X[ids[:nq]].shape} queries')

    # Save the following nb as base vectors, but first remove some duplicates. The final dataset size is slightly
    # smaller than 10M
    Xb = X[ids[nq:(nq + nb)]]
    if fname_rqa_10m_multiplicity is not None:
        Xb = rqa_10M_remove_duplicates(Xb, fname_rqa_10m_multiplicity)
    write_fvecs(f'{dataset_dir_out}/rqa_base_{int(nb / 1000000)}M.fvecs', Xb)
    print(f'Saving {Xb.shape} base vectors')

    # Also save a 1M subset of the base vectors for a smaller version of the dataset
    write_fvecs(f'{dataset_dir_out}/rqa_base_1M.fvecs', X[ids[nq:(nq + 1000000)]])
    print(f'Saving {X[ids[nq:(nq + 1000000)]].shape} base vectors')

    # Save the following 10k as learning set
    write_fvecs(f'{dataset_dir_out}/rqa_learn_{int(nl / 1000)}k.fvecs', X[ids[(nq + nb):(nq + nb + nl)]])
    print(f'Saving {X[ids[(nq + nb):(nq + nb + nl)]].shape} learning set')


def rqa_10M_remove_duplicates(X: npt.ArrayLike, fname_rqa_10m_multiplicity: str) -> npt.ArrayLike:
    norig = X.shape[0]

    print('Removing duplicates...')
    dup = toml.load(fname_rqa_10m_multiplicity)['conflict_groups']
    del_list = []
    for list in dup:
        for i, idx in enumerate(list):
            if i > 0:
                del_list.append(idx - 1)

    del_list = sorted(del_list)
    X = np.delete(X, del_list, axis=0)
    print(f'{norig - X.shape[0]} duplicated vectors were removed. Previous dataset size = {norig}, '
          f'new dataset size = {X.shape[0]}')

    return X


def split_OOD_query_learn_sets(init_file: int,
                               number_of_files: int,
                               fname_prefix_out: str,
                               dataset_dir_out: str):
    nq = 10000  # Number of queries
    nl = 10000  # Number of learning vectors

    # Load the generated embeddings file to split it into query and learning sets
    end_file = init_file + number_of_files
    fname_fvecs = f'{dataset_dir_out}/fvecs/{fname_prefix_out}_files{init_file}_{end_file}.fvecs'
    X = read_fvecs(fname_fvecs)

    Xq = X[:nq]
    Xlearn = X[nq:(nq + nl)]

    print(f'Saving OOD queries with shape: {Xq.shape}')
    write_fvecs(f'{dataset_dir_out}/rqa_query_{int(nq / 1000)}k_out_of_distro.fvecs', Xq)

    print(f'Saving OOD learning set with shape: {Xlearn.shape}')
    write_fvecs(f'{dataset_dir_out}/rqa_query_{int(nl / 1000)}k_out_of_distro_learn.fvecs', Xlearn)


def extract_queries(texts: list,
                    nRequired: Optional[int] = None,
                    questionRequired: Optional[bool] = False) -> list:
    np.random.seed(0)
    random.seed(0)

    if nRequired is None:
        nRequired = len(texts)

    assert nRequired <= len(texts), f'Cannot generate {nRequired} queries from {len(texts)} texts.'

    print("Extracting", nRequired, "queries from", len(texts), "texts.")

    # Shuffle the text ids so queries are chosen at random from the available texts
    nTexts = len(texts)
    ids = np.arange(nTexts)
    np.random.shuffle(ids)

    total_questions = 0
    queries = []
    i = 0
    while len(queries) < nRequired and i < nTexts:
        sent_text = nltk.sent_tokenize(texts[ids[i]])
        nSentences = len(sent_text)

        # If the current paragraph has at least one question mark, find
        # the sentences that have them and append them to the list of queries.
        if texts[ids[i]].find('?') > 0:
            for j in range(nSentences):
                if sent_text[j].find('?') > 0:
                    queries.append(sent_text[j])
                    total_questions += 1
        else:  # Otherwise, just choose one sentence from the paragraph at random.
            if not questionRequired:
                chosen_sent = random.sample(list(range(nSentences)), 1)[0]
                queries.append(sent_text[chosen_sent])
        i += 1

        if not i % 1000:
            print(len(queries), "queries appended so far, with", total_questions, "questions,",
                  100 * total_questions / len(queries))

    if len(queries) < nRequired:
        print(f'Warning: only {len(queries)} were extracted,from the {nRequired} requested')
    else:
        print(f'A total of {len(queries)} were extracted, {nRequired} were requested')

    return queries, ids


def write_ivecs(fname: str, m: npt.ArrayLike):
    n, d = m.shape
    m1 = np.empty((n, d + 1), dtype='int32')
    m1[:, 0] = d
    m1[:, 1:] = m
    m1.tofile(fname)


def write_fvecs(fname: str, m: npt.ArrayLike):
    m = m.astype('float32')
    write_ivecs(fname, m.view('int32'))


def read_ivecs(fname: str) -> npt.ArrayLike:
    x = np.fromfile(fname, dtype='int32')
    d = x[0]
    y = x.reshape(-1, d + 1)[:, 1:].copy()
    return y


def read_fvecs(fname: str) -> npt.ArrayLike:
    return read_ivecs(fname).view('float32')
