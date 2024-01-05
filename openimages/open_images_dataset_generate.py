import numpy as np
import numpy.typing as npt
from typing import Optional
import pandas as pd
import time
from PIL import Image
from transformers import AutoProcessor, CLIPProcessor, CLIPModel
import torch
import os


def generate_open_images_embeddings(batch_init_in: int,
                                    num_batches: int,
                                    batch_size: int,
                                    oi_base_dir: str,
                                    dataset_dir: str,
                                    dim: int,
                                    min_crop_size: int,
                                    verbose: Optional[bool] = False):
    f"""Generate CLIP embeddings from the Google's Open Images dataset save them in .fvecs format.

    We split the embedding generation for all image crops into batches of size batch_size.

    Keyword arguments:
    batch_init_in -- Identifier of the batch to start processing. The first processed image crop is the one at row
                    batch_init_in*batch_size in the data frame containing the metadata (after filtering for valid image
                    crops, see below).
    num_batches -- Number of batches of size batch_size to be processed, starting at batch_init_in.
    batch_size -- Number of image crops to process at a time.
    oi_base_dir -- Path to the location of the Google's Open Images images and metadata: 
                    - the images must be located at oi_base_dir/images/ 
                    - the metadata file is oi_base_dir/oidv6-train-annotations-bbox.csv
    dataset_dir --  The output embeddings will be saved at dataset_dir/fvecs/ and the metadata at dataset_dir/csv/                                
    dim -- CLIP model dimensionality
    min_crop_size -- Crops of size smaller than min_crop_size x min_crop_size will not be processed. Should 
                     be set to 0 as a minimum.
    verbose -- Optional; Set to True to print a warning message when an image crop is discarded because of its size. 
    """

    torch.set_grad_enabled(False)

    batch_end_in = batch_init_in + num_batches
    min_crop_size = max(int(min_crop_size), 0)  # Crops of size smaller than min_crop_size x min_crop_size will not be
    # processed. Should be set to 0 as a minimum.

    print(f'Processing: batch_init={batch_init_in}, num_batches={num_batches}, batch_sz={batch_size}, '
          f'crop_sz={min_crop_size}')

    model_str = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_str)
    processor = AutoProcessor.from_pretrained(model_str)

    # Load the image ids and the corresponding classes. Each image id might have multiple classes.
    annotations_fname = f'{oi_base_dir}/oidv6-train-annotations-bbox.csv'
    print('Loading annotations csv data frame')
    df_annotations = pd.read_csv(annotations_fname)
    print('Finished loading annotations csv data frame')
    df_annotations = df_annotations.query("Confidence == 1")  # Take only labels of objects present in the image
    df_annotations = df_annotations.reset_index()

    # Keep only the subpart of the data frame that will be used to generate the embeddings in this set of batches
    df_annotations = df_annotations.loc[batch_init_in * batch_size:(batch_end_in * batch_size - 1)].reset_index(
        drop=True)
    n_annotations = len(df_annotations)
    print(f'Metadata data frame len = {n_annotations}')

    df_annotations['crop_size_x'] = -1
    df_annotations['crop_size_y'] = -1
    df_annotations['crop_xmin'] = -1
    df_annotations['crop_xmax'] = -1
    df_annotations['crop_ymin'] = -1
    df_annotations['crop_ymax'] = -1
    df_annotations['discarded'] = 0
    df_annotations['embd_id'] = -1

    num_batches = int(np.round(n_annotations / batch_size))
    print(f'A maximum of {n_annotations} embeddings can be created, so the num_batches was set to {num_batches}.')

    fname_fvecs = f'{dataset_dir}/fvecs/oi_clip_emb_d{dim}_mc{min_crop_size}_boxed_b{batch_init_in}_{num_batches}.fvecs'
    fname_dframe = f'{dataset_dir}/csv/oi_clip_emb_d{dim}_mc{min_crop_size}_boxed_b{batch_init_in}_{num_batches}.csv'

    if not os.path.exists(f'{dataset_dir}/fvecs'):
        os.makedirs(f'{dataset_dir}/fvecs')
    if not os.path.exists(f'{dataset_dir}/csv'):
        os.makedirs(f'{dataset_dir}/csv')

    embeddings = np.zeros([n_annotations, dim])

    device = "cuda:" + str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
    print('Using', device)
    model = model.to(device)

    model.eval()

    t0 = time.time()
    current_img_id = ""
    total_embeddings = 0
    discarded_crops = 0
    embd_id = 0
    for batch in range(num_batches):

        batch_init = batch * batch_size
        batch_end = np.min([batch_init + batch_size, n_annotations])

        # First prepare list of images for the current batch of embeddings
        image_list = []
        eff_batch_size = 0
        for ii in range(batch_init, batch_end):

            img_id = df_annotations.loc[ii, 'ImageID']

            # If necessary reload the image, otherwise, keep extracting boxes from the same image
            if img_id != current_img_id:
                current_img_id = img_id
                image_fname = f'{oi_base_dir}/images/{img_id}.jpg'
                image = Image.open(image_fname)

            xmin = int(image.size[0] * df_annotations.loc[ii, 'XMin'])
            xmax = int(image.size[0] * df_annotations.loc[ii, 'XMax'])
            ymin = int(image.size[1] * df_annotations.loc[ii, 'YMin'])
            ymax = int(image.size[1] * df_annotations.loc[ii, 'YMax'])

            crop_size = (xmax - xmin, ymax - ymin)

            df_annotations.loc[ii, 'crop_size_x'] = crop_size[0]
            df_annotations.loc[ii, 'crop_size_y'] = crop_size[1]
            df_annotations.loc[ii, 'crop_xmin'] = xmin
            df_annotations.loc[ii, 'crop_xmax'] = xmax
            df_annotations.loc[ii, 'crop_ymin'] = ymin
            df_annotations.loc[ii, 'crop_ymax'] = ymax

            if crop_size[0] < min_crop_size or crop_size[1] < min_crop_size:
                discarded_crops += 1
                df_annotations.loc[ii, 'discarded'] = 1
                if verbose:
                    print(
                        f'Warning: The {ii}-th crop, image {img_id} {image.size} with coordinates '
                        f'({xmin},{xmax},{ymin},{ymax}) and size {crop_size} was discarded.')
            else:
                image_list.append(image.crop((xmin, ymin, xmax, ymax)))
                df_annotations.loc[ii, 'embd_id'] = embd_id
                embd_id += 1
                eff_batch_size += 1

        if len(image_list) > 0:

            print(f'Pre-processing {eff_batch_size} images')
            inputs = processor(images=image_list, return_tensors="pt")
            print('Finished pre-processing images and sending to device')
            inputs = inputs.to(device)
            print('Finished sending to device')

            with torch.no_grad():
                print(f'Generating embeddings for batch {batch} with size {eff_batch_size}')
                outputs = model.get_image_features(**inputs)
                print('Finished generating embeddings')

                print('Assigning embeddings')
                embeddings[total_embeddings:(total_embeddings + eff_batch_size)] = outputs.squeeze(
                    0).cpu().detach().numpy()
                print('Finished assigning embeddings')
                del outputs

            total_embeddings += eff_batch_size
            print(
                f'Batch {batch}/{num_batches}: {100 * (total_embeddings + discarded_crops) / n_annotations:.4f} '
                f'% of embeddings completed, {total_embeddings} ({100 * total_embeddings / ii:.2f} %) generated '
                f'and {discarded_crops} ({100 * discarded_crops / ii:.2f} %) discarded in {(time.time() - t0):.3f} s, '
                f'ii: {ii}, eff_bsize: {eff_batch_size}, emb_id: {embd_id}')

            if batch > 0 and not batch % int(num_batches / 8):
                t1 = time.time()
                print('Saving intermediate results')
                df_annotations.to_csv(fname_dframe)
                write_fvecs(fname_fvecs, embeddings[:total_embeddings])
                t2 = time.time()
                print(f'Saving intermediate results took {t2 - t1:.5f} s')
        else:
            print(
                f'Batch {batch}/{num_batches}: Warning: Empty image list!')

    print(f'A total of {discarded_crops} crops were discarded.')

    print(f'Saving {total_embeddings} embeddings to .fvecs format.')
    write_fvecs(fname_fvecs, embeddings[:total_embeddings])

    print('Saving data frame to csv')
    df_annotations.to_csv(fname_dframe)
    print('Finished saving data.')

    # Sanity check
    assert len(df_annotations) == embeddings[:total_embeddings].shape[
        0] + discarded_crops, f"The csv number of entries " \
                              f"{len(df_annotations)} does" \
                              f"not match the expected value" \
                              f"{embeddings[:total_embeddings].shape[0] + discarded_crops}" \
                              f"({embeddings[:total_embeddings].shape[0]} embeddings and " \
                              f"{discarded_crops} discarded crops)"


def merge_embedding_files(dataset_dir: str,
                          dim: int,
                          min_crop_size: int,
                          num_batches: int,
                          num_batches_last: int,
                          batch_init_in_vals: list) -> (npt.ArrayLike, pd.DataFrame):
    f""" Merge the embeddings from .fvecs files generated using function generate_open_images_embeddings. The files 
         to be merged are named: 
         
         [dataset_dir]/fvecs/oi_clip_emb_d[dim]_mc[min_crop_size]_boxed_b[batch_init_in]_[num_batches].fvecs'

    Keyword arguments:        
    dataset_dir -- Path to the location of .fvecs files generated using generate_open_images_embeddings. The files must be
                located at dataset_dir/fvecs.
    dim -- Same dim used by generate_open_images_embeddings    
    min_crop_size -- Same min_crop_size used by generate_open_images_embeddings
    num_batches -- Same num_batches used by generate_open_images_embeddings
    num_batches_last -- The last file to be merged may have a number of batches smaller than num_batches, specify the 
                        correct value in order to be able to load that file.                       
    batch_init_in -- List of the batch_init_in values used by generate_open_images_embeddings.
    """

    assert len(batch_init_in_vals) >= 1, 'Need to specify at least one batch_init values.'

    if not os.path.exists(f'{dataset_dir}/fvecs'):
        os.makedirs(f'{dataset_dir}/fvecs')
    if not os.path.exists(f'{dataset_dir}/csv'):
        os.makedirs(f'{dataset_dir}/csv')

    # Load initial batch
    batch_init_in = batch_init_in_vals[0]
    fname_fvecs = f'{dataset_dir}/fvecs/oi_clip_emb_d{dim}_mc{min_crop_size}_boxed_b{batch_init_in}_{num_batches}.fvecs'
    fname_dframe = f'{dataset_dir}/csv/oi_clip_emb_d{dim}_mc{min_crop_size}_boxed_b{batch_init_in}_{num_batches}.csv'
    X_merged = read_fvecs(fname_fvecs)
    df_merged = pd.read_csv(fname_dframe)
    df_merged = df_merged.query("discarded == 0")

    print(f'Loading {X_merged.shape[0]} from file {fname_fvecs}')
    assert len(
        df_merged) == X_merged.shape[0], f'Embeddings and metadata size mismatch: {len(df_merged)} vs {X_merged.shape}'

    if len(batch_init_in_vals) > 2:
        # Continue loading the following batches
        for batch_init_in in batch_init_in_vals[1:-1]:
            fname_fvecs = f'{dataset_dir}/fvecs/oi_clip_emb_d{dim}_mc{min_crop_size}_boxed_b{batch_init_in}_{num_batches}.fvecs'
            fname_dframe = f'{dataset_dir}/csv/oi_clip_emb_d{dim}_mc{min_crop_size}_boxed_b{batch_init_in}_{num_batches}.csv'

            X = read_fvecs(fname_fvecs)
            X_merged = np.concatenate([X_merged, X])

            print(f'Loaded {X.shape[0]} embeddings from file {fname_fvecs}, total embeddings = {X_merged.shape[0]}')

            df = pd.read_csv(fname_dframe)
            df = df.query("discarded == 0")
            df_merged = pd.concat([df_merged, df])

            assert len(df_merged) == X_merged.shape[0], f'Embeddings and metadata size mismatch: {len(df_merged)} vs ' \
                                                        f'{X_merged.shape}'

    if len(batch_init_in_vals) >= 2:
        # Load final batch
        batch_init_in = batch_init_in_vals[-1]
        fname_fvecs = f'{dataset_dir}/fvecs/oi_clip_emb_d{dim}_mc{min_crop_size}_boxed_b{batch_init_in}_{num_batches_last}.fvecs'
        fname_dframe = f'{dataset_dir}/csv/oi_clip_emb_d{dim}_mc{min_crop_size}_boxed_b{batch_init_in}_{num_batches_last}.csv'
        X = read_fvecs(fname_fvecs)
        X_merged = np.concatenate([X_merged, X])
        print(f'Loaded {X.shape[0]} embeddings from file {fname_fvecs}, total embeddings = {X_merged.shape[0]}')

        df = pd.read_csv(fname_dframe)
        df = df.query("discarded == 0")
        df_merged = pd.concat([df_merged, df])

        assert len(df_merged) == X_merged.shape[0], f'Embeddings and metadata size mismatch: {len(df_merged)} vs ' \
                                                    f'{X_merged.shape}'

    # Dropping this embedding id because it corresponds to an old reference no longer useful.
    # A new EmbeddingID column will be created.
    df_merged = df_merged.drop(columns=['embd_id'])
    df_merged['EmbeddingID'] = list(range(X_merged.shape[0]))
    df_merged = df_merged.reset_index()

    # Normalizing embeddings
    print('Generating and checking normalized version   ')
    X_merged[np.linalg.norm(X_merged, axis=1) == 0] = 1.0 / np.sqrt(X_merged.shape[1])
    X_merged /= np.linalg.norm(X_merged, axis=1)[:, np.newaxis]

    return X_merged, df_merged


def split_base_query_learn_sets(X_merged: npt.ArrayLike,
                                df_merged: pd.DataFrame,
                                dataset_dir: str,
                                dim: int,
                                min_crop_size: int):
    f""" Split the generated embeddings into base, query and learning sets and save the corresponding files. Two
         sets of base vectors are created, one containing 13M and the other containing 1M vectors. The query and
         learning sets contain 10k vectors each.

    Keyword arguments:        
    X_merged -- Embeddings to be split.
    df_merged -- Data frame containing the metadata corresponding to the embeddings.    
    dataset_dir -- Directory where the output dataset files will be saved.
    dim -- Same num_batches used by generate_open_images_embeddings
    min_crop_size -- Same min_crop_size used by generate_open_images_embeddings 

    """

    seed = 1234
    np.random.seed(seed)

    nb = 13000000
    nq = 10000
    nl = 10000

    fname_shuff_ids = f'{dataset_dir}/fvecs/oi_clip_emb_d{dim}_mc{min_crop_size}_boxed_shuff_ids.npz'
    fname_fvecs_base = f'{dataset_dir}/fvecs/oi_clip_emb_d{dim}_mc{min_crop_size}_boxed_base_{int(nb / 1000000)}M.fvecs'
    fname_fvecs_base1M = f'{dataset_dir}/fvecs/oi_clip_emb_d{dim}_mc{min_crop_size}_boxed_base_1M.fvecs'
    fname_fvecs_queries = f'{dataset_dir}/fvecs/oi_clip_emb_d{dim}_mc{min_crop_size}_boxed_queries_{int(nq / 1000)}k.fvecs'
    fname_fvecs_learn = f'{dataset_dir}/fvecs/oi_clip_emb_d{dim}_mc{min_crop_size}_boxed_learn_{int(nl / 1000)}k.fvecs'

    fname_dframe_base = f'{dataset_dir}/csv/oi_clip_emb_d{dim}_mc{min_crop_size}_boxed_base_{int(nb / 1000000)}M.csv'
    fname_dframe_base1M = f'{dataset_dir}/csv/oi_clip_emb_d{dim}_mc{min_crop_size}_boxed_base_1M.csv'
    fname_dframe_queries = f'{dataset_dir}/csv/oi_clip_emb_d{dim}_mc{min_crop_size}_boxed_queries_{int(nq / 1000)}k.csv'
    fname_dframe_learn = f'{dataset_dir}/csv/oi_clip_emb_d{dim}_mc{min_crop_size}_boxed_learn_{int(nl / 1000)}k.csv'

    # Split data into base vectors and queries, save ids for split and csv files
    print('Shuffling ids')
    all_ids = np.arange(X_merged.shape[0])
    np.random.shuffle(all_ids)

    print('Saving shuffled ids')
    np.savez(fname_shuff_ids, all_ids=all_ids)

    # Saving query set
    print(
        f'Saving {X_merged[all_ids[:nq]].shape[0]} queries vectors with {X_merged[all_ids[:nq]].shape[1]} dimensions.')
    write_fvecs(fname_fvecs_queries, X_merged[all_ids[:nq]])
    print(f'Saving queries csv with size {len(df_merged.loc[all_ids[:nq]])}')
    df_merged.loc[all_ids[:nq]].to_csv(fname_dframe_queries)

    # Saving base vectors
    print(f'Saving base vectors with size {X_merged[all_ids[nq:(nq + nb)]].shape}...')
    write_fvecs(fname_fvecs_base, X_merged[all_ids[nq:(nq + nb)]])
    print(f'Saving base csv with size {len(df_merged.loc[all_ids[nq:(nq + nb)]])}')
    df_merged.loc[all_ids[nq:(nq + nb)]].to_csv(fname_dframe_base)

    # Saving learning set
    print(f'Saving learn vectors with size {X_merged[all_ids[(nq + nb):(nq + nb + nl)]].shape}...')
    write_fvecs(fname_fvecs_learn, X_merged[all_ids[(nq + nb):(nq + nb + nl)]])
    print(f'Saving base csv with size {len(df_merged.loc[all_ids[(nq + nb):(nq + nb + nl)]])}')
    df_merged.loc[all_ids[(nq + nb):(nq + nb + nl)]].to_csv(fname_dframe_learn)

    # Saving the 1M version of the base vectors
    print(f'Saving base vectors with size {X_merged[all_ids[nq:(nq + 1000000)]].shape}...')
    write_fvecs(fname_fvecs_base1M, X_merged[all_ids[nq:(nq + 1000000)]])
    print(f'Saving base csv with size {len(df_merged.loc[all_ids[nq:(nq + 1000000)]])}')
    df_merged.loc[all_ids[nq:(nq + 1000000)]].to_csv(fname_dframe_base1M)


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
