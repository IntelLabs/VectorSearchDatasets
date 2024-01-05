# wit-512-1M Dataset Generator

This repository provides code to generate base and query (test and learn sets) embeddings for similarity search benchmarking
and evaluation on high-dimensional vectors. The dataset is designed to benchmark similarity search methods under
scenarios with out-of-distribution (OOD) queries stemming from a text-to-image application [[1]](#1).

The WIT dataset[[2]](#2) is a multimodal multilingual dataset that contains 37 million rich image-text examples
extracted from Wikipedia pages. For each example in the first million training images
(downloaded from [here](https://storage.cloud.google.com/wikimedia-image-caption-public/image_data_train.tar)), we take the image and encode it
using the multimodal OpenAI CLIP-ViT-B32 model [[3]](#3) to generate a database vector.
We create the query set using the first 20K text descriptions in one of the provided test sets (concatenating
the Reference and Attribution description fields) and generating the corresponding embeddings using CLIPViT-B32-multilingual-v1 [[4]](#4).
The use of CLIP-ViT-B32 for images and multi-lingual CLIP-ViT-B32-multilingual-v1 for text follows the protocol suggested
[here]( https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1).
Finally, for each query, we compute the 100 ground truth nearest neighbors using maximum inner product.
We use the first 10K queries as a query test set and the remaining 10K as a learn set.

The metric for similarity search used with this dataset is inner product.


## Steps to generate the dataset

The script [wit_dataset_1M.py](wit_dataset_1M.py) generates 1 million base vectors from the provided training images
and two sets of queries, test and learn set, each with 10K vectors from the text descriptions of the provided test set.

Here is a summary of the **steps to generate this dataset**:

1. **Download the WIT training Images and test set**.
We download the training images from [here](https://storage.cloud.google.com/wikimedia-image-caption-public/image_data_train.tar) and extract them in the desired location
> **_NOTE:_** the above link requires Google login authentication to download the training images.

```
tar -xvf image_data_train.tar -C $BASE_PATH
```
The extracted files will be in $BASE_PATH/image_data_train/image_pixels/.
The images are encoded in base64 format, see image/file format details [here](https://www.kaggle.com/c/wikipedia-image-caption/data).

For queries, we download one of the [test set](https://storage.googleapis.com/gresearch/wit/wit_v1.test.all-00000-of-00005.tsv.gz) and extract it
```
mkdir -p $BASE_PATH/test_set
tar -xvfz wit_v1.test.all-00000-of-00005.tsv.gz -C $BASE_PATH/test_set
```

2. **Run** the `wit_dataset_1M.py` script to generate `.fvecs` files containing the base
   , query and learn set vectors. **Remember to set the path** to the folder where the
   downloaded training images, test files are located.

3. **Generate the ground-truth** by conducting an exhaustive search with the inner product metric.
   We provide the ground-truth files for the query test and learn sets,

4. Functions `read_fvecs` and `read_ivecs` can be used to read `.fvecs` and `.ivecs` files respectively.

> **_NOTE:_**  Due to floating-point arithmetic precision the vector embeddings generated using the provided
> code in different machines may slightly vary. Keep in mind that this could cause small discrepancies with the provided ground-truth.


## References
Reference to cite when you use datasets generated with this code in a research paper:

```
@article{tepper2023leanvec,
        title={LeanVec: Search your vectors faster by making them fit},
      	author={Mariano Tepper and Ishwar Singh Bhati and Cecilia Aguerrebere and Mark Hildebrand and Ted Willke},
        year={2023},
      	journal={arXiv},
      	doi={https://doi.org/10.48550/arXiv.2312.16335}
}
```
<a id="1">[1]</a>
Mariano Tepper, Ishwar Singh Bhati, Cecilia Aguerrebere, Mark Hildebrand, and Ted Willke:
LeanVec: Search your vectors faster by making them fit. (2023)

<a id="2">[2]</a>
Krishna Srinivasan, Karthik Raman, Jiecao Chen, Michael Bendersky, and Marc Najork:
WIT: Wikipedia-based Image Text Dataset for Multimodal Multilingual Machine Learning. (2021)

<a id="3">[3]</a>
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever:
Learning Transferable Visual Models From Natural Language Supervision. (2021)

<a id="4">[4]</a>
Nils Reimers, and Iryna Gurevych:
Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation. (2020)

This "research quality code"  is for Non-Commercial purposes provided by Intel "As Is" without any express or implied
warranty of any kind. Please see the dataset's applicable license for terms and conditions. Intel does not own the
rights to this data set and does not confer any rights to it. Intel does not warrant or assume responsibility for the accuracy or completeness of any information, text, graphics, links or other items within the code. A thorough security review has not been performed on this code. Additionally, this repository may contain components that are out of date or contain known security vulnerabilities.

