from sentence_transformers import SentenceTransformer, util
from PIL import Image, ImageFile
import glob
import os
import sys
import csv
import io, base64
import gzip
import warnings
import numpy as np
import numpy.typing as npt
from natsort import natsorted
from typing import Optional


def generate_image_embeddings(images_dir: str, n_vecs: int, output_dir: str, fname_prefix: str):
    f"""Generate image embeddings using OpneAI CLIP-ViT-B32 model and save them in .fvecs format.

    Keyword arguments:
    images_dir   -- Input dir with *.csv.gz files storing images in base64 encoded bytes.
                    See  file/image format details here: https://www.kaggle.com/c/wikipedia-image-caption/data
                    Files are accessed in naturally sorted order
    n_vecs       -- Number of embedding to generate
    output_dir  -- The output embeddings will be saved at output_dir/embeddings
    fname_prefix -- Name of the output file: fname_prefix.fvec
    """

    # Use the original clip-ViT-B-32 for encoding images
    img_model = SentenceTransformer('clip-ViT-B-32')

    filenames = glob.glob(f'{images_dir}/*.csv.gz')
    # Sorted based on file names
    filenames = natsorted(filenames)

    num_files = len(filenames)
    csv.field_size_limit(sys.maxsize)
    emb_dims = 512

    img_embeddings = np.empty((0, emb_dims), np.float32)
    total_images = 0
    for i in range(0, num_files):
        images = []
        with gzip.open(filenames[i], 'rt', newline='') as file:
            csv_reader = csv.reader(file, delimiter = '\t')
            print(f'Reading file {filenames[i]}')
            for row in csv_reader:
                img = Image.open(io.BytesIO(base64.decodebytes(bytes(row[1], "utf-8"))))
                img = img.convert('RGBA')
                images.append(img)
                total_images = total_images + 1
                if total_images == n_vecs:
                    break

            ls_embeddings = img_model.encode(images)
            assert ls_embeddings.shape[1] == emb_dims
            img_embeddings = np.append(img_embeddings, ls_embeddings, axis = 0)
            print(f'Generated {img_embeddings.shape[0]} image embeddings')

            if img_embeddings.shape[0] >= n_vecs:
                break

    if (img_embeddings.shape[0] < n_vecs):
        warnings.warn(f'Insufficient images to generate {n_vecs} embeddings. Generated {img_embeddings.shape[0]}')

    if not os.path.exists(f'{output_dir}/embeddings'):
        os.makedirs(f'{output_dir}/embeddings')
    output_file = f'{output_dir}/embeddings/{fname_prefix}.fvecs'
    print(f'Saving in {output_file}')
    write_fvecs(img_embeddings[:n_vecs], output_file)


def generate_text_embeddings(fname: str, n_vecs: int, output_dir: str, fname_prefix: str, offset: Optional[int] = 0):
    f"""Generate text embeddings using OpneAI CLIP-ViT-B32-multilingual-v1 model and save them in .fvecs format.

    Keyword arguments:
    fname        -- Input .tsv file storing test data set in WIT format,
                    see format details here https://github.com/google-research-datasets/wit/blob/main/DATA.md
    n_vecs       -- Number of embedding to generate
    output_dir  -- The output embeddings will be saved at output_dir/embeddings
    fname_prefix -- Name of the output file: fname_prefix.fvec
    offset       -- Optional; Generate embeddings after skipping these many entries
    """


    # Use the multi-lingual model for text embeddings
    text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
    emb_dims = 512

    results = []
    total_texts = 0
    with open(fname) as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for count, row in enumerate(rd):
            # Skip the first row, it contains field descriptions
            if count > offset:
                results.append(row)
                total_texts = total_texts + 1
                if total_texts == n_vecs:
                    break


    num_rows = len(results)

    # Create texts by concatenating the Reference and Attribution fields
    texts = []
    for row in range(0, num_rows):
        texts.append(results[row][6] + results[row][7])

    text_embeddings = text_model.encode(texts)
    assert text_embeddings.shape[1] == emb_dims
    print(f'Generated {text_embeddings.shape[0]} text embeddings')

    if (text_embeddings.shape[0] < n_vecs):
        warnings.warn(f'Insufficient text/entries to generate {n_vecs} embeddings. Generated {text_embeddings.shape[0]}')

    if not os.path.exists(f'{output_dir}/embeddings'):
        os.makedirs(f'{output_dir}/embeddings')
    output_file = f'{output_dir}/embeddings/{fname_prefix}.fvecs'
    print(f'Saving in {output_file}')
    write_fvecs(text_embeddings[:n_vecs], output_file)


def write_ivecs(m: npt.ArrayLike, fname: str):
    n, d = m.shape
    m1 = np.empty((n, d + 1), dtype='int32')
    m1[:, 0] = d
    m1[:, 1:] = m
    m1.tofile(fname)


def write_fvecs(m: npt.ArrayLike, fname: str):
    m = m.astype('float32')
    write_ivecs(m.view('int32'), fname)


def read_ivecs(fname: str) -> npt.ArrayLike:
    x = np.fromfile(fname, dtype='int32')
    d = x[0]
    y = x.reshape(-1, d + 1)[:, 1:].copy()
    return y


def read_fvecs(fname: str) -> npt.ArrayLike:
    return read_ivecs(fname).view('float32')


