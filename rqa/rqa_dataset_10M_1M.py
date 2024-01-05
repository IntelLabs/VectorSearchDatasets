import numpy as np
import random
import rocketqa
from rqa_dataset_generate import generate_rqa_embeddings, merge_embedding_files, split_base_query_learn_sets, \
    split_OOD_query_learn_sets

if __name__ == "__main__":
    # This is an example script to generate the rqa-768-10M and rqa-768-1M datasets, containing:
    #     --- 10M and 1M base vector embeddings
    #     --- in-distribution (ID) query and learning sets (10k elements each)
    #     --- out-of-distribution (OOD) query and learning sets (10k elements each)
    #
    # introduced in the paper "LeanVec: Search your vectors faster by making them fit", Tepper, M.; Bhati I.;
    # Aguerrebere, C.; Hildebrand M.; Tepper M.; Willke T.
    #
    # Please see the documentation of the generate_rqa_embeddings function for details on the required parameters.
    #
    # The dataset was generated using text snippets from the files in the "en" (305GB) variant of the C4 dataset
    # available at: https://huggingface.co/datasets/allenai/c4
    # We used files c4-train.00000-of-01024.json.gz to c4-train.00032-of-01024.json.gz in the train folder (c4/en/train)
    # to generate the base vectors and the ID query and learning sets. We used file c4-validation.00000-of-00008.json
    # in the validation folder (c4/en/validation/) to generate the OOD query and learning sets.
    #

    np.random.seed(0)
    random.seed(0)

    base_C4_dir = '/home/user/research/datasets/c4/en/'  # Set this path to where your c4/en folder is located
    dataset_dir_out = '/home/user/research/datasets/rqa-final/'        # Set this path to where the final rqa dataset
                                                                       # files will be saved
    cache_folder = f'/home/user/.cache/huggingface/datasets/'  # Set to the hugginface datasets cache path

    # Load dual encoder model
    print('Loading rocketqa model...')
    batch_size = 512
    dual_encoder = rocketqa.load_model(model="v1_marco_de", use_cuda=True, device_id=0, batch_size=batch_size)

    ##
    ## --- Base vectors and IID query learning sets generation ----
    ##
    c4_dataset_dir = f'{base_C4_dir}/train/'
    fname_prefix_out = 'c4-en-base-vectors'

    init_file = 0              # id of the initial C4 text file to use
    total_files = 32           # total number of C4 files to process starting at init_file
    number_of_files_batch = 4  # total files per batch to process (we process the total_files in parallel to speed-up
                               # the computation, see details below)

    # Generating 10M embeddings requires 32 files. We split the generation in batches of 4 files, which we run
    # in multiple servers in parallel to speed-up the process. Each of these runs will generate a .fvecs file with the
    # embeddings corresponding to those 4 files. The following for loop shows the exact runs that we launched in
    # parallel.
    for init_file_partial in range(init_file, total_files, number_of_files_batch):
        # Generate embeddings from files init_file_partial to init_file_partial+number_of_files_batch and save them to
        # a .fvecs file in dataset_dir/fvecs/ folder.
        generate_rqa_embeddings(dual_encoder, init_file_partial, number_of_files_batch, c4_dataset_dir,
                                 dataset_dir_out, fname_prefix_out, cache_folder)

    # Merge the 8 .fvecs files containing all the generated embeddings from the 32 text input files from the C4 dataset.
    final_file = total_files - number_of_files_batch
    X = merge_embedding_files(init_file, number_of_files_batch, final_file, dataset_dir_out, fname_prefix_out)

    # Finally split the embeddings into base, query and learning IID sets and save them to dataset_dir_out
    fname_rqa_10m_multiplicity = './rqa_10m_multiplicity.toml'
    split_base_query_learn_sets(X, init_file, final_file, fname_prefix_out, dataset_dir_out,
                          fname_rqa_10m_multiplicity=fname_rqa_10m_multiplicity)

    #
    # --- Out-of-distribution (OOD) query and learning vectors generation ----
    #
    # Generate OOD embeddings and save them to a .fvecs file in dataset_dir/embeddings. The function fvecs_read() in
    # rqa_dataset_generate.py can be used to read the .fvecs files.
    c4_dataset_dir = f'{base_C4_dir}/validation/'
    fname_prefix_out = 'c4-en-query-vectors'
    init_file = 0
    number_of_files = 1  # Make sure the input files (1 in this case) are enough to generate the requested number of
                         # embeddings if num_embd is specified. A warning message will be printed otherwise.
    num_embd = 20000
    generate_rqa_embeddings(dual_encoder, init_file, number_of_files, c4_dataset_dir, dataset_dir_out,
                                fname_prefix_out, cache_folder,
                                num_embd=num_embd, generate_queries=True, questionRequired=True)

    # Split the generated embeddings into the query and learning sets and save the corresponding .fvecs files
    split_OOD_query_learn_sets(init_file, number_of_files, fname_prefix_out, dataset_dir_out)