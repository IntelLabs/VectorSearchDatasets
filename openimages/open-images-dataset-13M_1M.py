from open_images_dataset_generate import generate_open_images_embeddings, merge_embedding_files, \
    split_base_query_learn_sets

if __name__ == "__main__":
    # This is an example script to generate the open-images-512-10M and open-images-512-1M datasets, containing:
    #
    #     --- 13M and 1M base vector embeddings, respectively
    #     --- query and learning sets (10k elements each)
    #
    # introduced in the paper "Locally-adaptive Quantization for Streaming Vector Search", Aguerrebere, C.;
    # Hildebrand M.; Bhati I.; Willke T; Tepper M.
    #
    #
    # The dataset consists of CLIP embeddings generated using image crops obtained from the Google's Open Images dataset.
    # We take the 1.9M images subset that includes dense annotations and use the provided bounding boxes to extract
    # over 13M image crops with their corresponding class labels. There are a total of 434 classes representing
    # diverse objects. See the README for more details.
    #
    # Please see the documentation of the generate_open_images_embeddings function for details on the required
    # parameters.

    oi_base_dir = '/export/data/user/research/datasets/open-images/'  # Path to the location of the Google's Open
    # Images images and metadata:
    # - the images must be located at
    #    oi_base_dir/images/
    # - the metadata file is
    # oi_base_dir/oidv6-train-annotations-bbox.csv

    dataset_dir = '/export/data/user/research/datasets/open-images/final-dataset/'  # Folder where the created
    # dataset will be saved

    batch_init_in_vals = list(range(0, 14268, 714))
    batch_size = 1024
    num_batches = 714
    dim = 512
    min_crop_size = 16
    num_batches_last = 702

    # We split the image crops into batches, which we run in multiple servers in parallel to speed-up the process.
    # Each of these runs will generate a .fvecs file with the embeddings corresponding to the given batch.
    # The following for loop shows the exact runs that we launched in parallel. We recommend to run this for loop as
    # independent runs and not sequentially to speed-up the process.
    for batch_init_in in batch_init_in_vals:
        # Generate embeddings for the given batch of image crops and save them to
        # [dataset_dir]/fvecs/oi_clip_emb_d[dim]_mc[min_crop_size]_boxed_b[batch_init_in]_[num_batches].fvecs'
        generate_open_images_embeddings(batch_init_in,
                                        num_batches,
                                        batch_size,
                                        oi_base_dir,
                                        dataset_dir,
                                        dim,
                                        min_crop_size)

    # Merge the .fvecs files into a single numpy array containing all the embeddings
    X_merged, df_merged = merge_embedding_files(dataset_dir,
                                                dim,
                                                min_crop_size,
                                                num_batches,
                                                num_batches_last,
                                                batch_init_in_vals)

    # Split into base, query and learning sets and save
    split_base_query_learn_sets(X_merged,
                                df_merged,
                                dataset_dir,
                                dim,
                                min_crop_size)
