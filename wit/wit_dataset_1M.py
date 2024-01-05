from wit_dataset_generate import  generate_image_embeddings, generate_text_embeddings


if __name__ == "__main__":
    # This is an example script to generate the wit-512-1M dataset, containing:
    #
    #     --- 1M base vectors from image embeddings
    #     --- out-of-distribution (OOD) query and learning sets (10k vectors each) from text embeddings
    #
    # Introduced in the paper ""LeanVec: Search your vectors faster by making them fit", 2023,
    # Tepper, Bhati, Aguerrebere, Hildebrand, Willke (https://arxiv.org/abs/2312.16335)
    #
    #
    # This dataset is created using a subset of Google's multimodal multilingual WIT dataset,
    # using image-text examples extracted from Wikipedia pages (https://github.com/google-research-datasets/wit).
    # To generate a base vector, we take the image and encode it using OpenAI CLIP-ViT-B32 model.
    # For queries, we use text descriptions in one of the provided test sets
    # (concatenating the Reference and Attribution description fields) and generating the corresponding
    # embeddings using OpenAI CLIP-ViT-B32-multilingual-v1. We followed the steps suggested in
    # https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1
    #
    # See the README for more details.
    #
    # Please see the documentation of the generate_image_embeddings and generate_text_embeddings functions
    # for details on the required parameters.


    base_path = '/raid0/ishwarsi/datasets/wit'  # Path to the location of the WIT Images/Test datasets are located
    images_dir = f'{base_path}/image_data_train/image_pixels' # Images files directory storing *.csv.gz files
    output_dir = f'{base_path}/output' # Directory where the created dataset will be saved

    # Files saved in [output_dir]/embeddings/{fname_prefix}.fvecs'

    # Generate image embeddings from the images provided in the images_dir
    fname_prefix = 'wit_base_1M'
    num_vecs = 1000_000
    generate_image_embeddings(images_dir, num_vecs, output_dir, fname_prefix)

    test_file = f'{base_path}/test_set/wit_v1.test.all-00000-of-00005.tsv'

    # Generate text embeddings from the test file
    fname_prefix = 'wit_query_10k'
    num_vecs = 10_000
    generate_text_embeddings(test_file, num_vecs, output_dir, fname_prefix)

    # Learn queries start from an offset (the last parameter)
    fname_prefix = 'wit_learn_query_10k'
    num_vecs = 10_000
    num_skip = 10_000
    generate_text_embeddings(test_file, num_vecs, output_dir, fname_prefix, num_skip)
