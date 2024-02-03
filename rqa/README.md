# rqa-768-10M Dataset Generator

This repository provides code to generate base, query and learning vectors for similarity
search benchmarking and evaluation on high-dimensional vectors stemming from large language models.
The dataset is designed to benchmark similarity search methods under scenarios with out-of-distribution
queries (in-distribution queries are also provided). 

We use the dense passage retriever model RocketQA [[1]](#1), to encode text snippets from the C4 dataset [[2]](#2) to 
generate 768-dimensional vectors. The dataset contains:

- base vector embeddings (10M and 1M versions)
- in-distribution (ID) query and learning sets (10k elements each)
- out-of-distribution (OOD) query and learning sets (10k elements each)

The base vectors and the ID query and learning sets are generated using RocketQA paragraph 
encoder, whereas the RocketQA query encoder is used for the OOD query and learning sets. 

The metric for similarity search is inner product [[1]](#1). 

We provide the [script](rqa_dataset_10M_1M.py) to generate the 10M and 1M versions of the dataset 
introduced in [[3]](#3), but larger versions can be created by using the
`generate_rqa_embeddings` function to process more C4 text files 
(see [rqa_dataset_10M_1M.py](rqa_dataset_10M_1M.py) for an example).

## Steps to generate the dataset

Use the script [rqa_dataset_10M_1M.py](rqa_dataset_10M_1M.py) to 
generate the datasets rqa-768-10M and rqa-768-1M introduced in [[3]](#3). The corresponding ground-truth 
(available [here](groundtruth)) is generated conducting an exhaustive search with the inner 
product metric.

Here is a summary of the **steps to generate this dataset**:

1. **Download the files** corresponding to the `en` variant of the C4 dataset accesible [here](https://huggingface.co/datasets/allenai/c4). 
The complete set of files requires 350GB of storage, so you might want to follow the instructions to download only a subset. 
   
To generate:
   
   - the 10M embeddings and ID query and learning sets we used the first 32 files from the train set (i.e., files `c4-train.00000-of-01024.json.gz` to `c4-train.00032-of-01024.json.gz` in `c4/en/train`).
   - the OOD query and learning sets we used the first file form the validation set (i.e., `c4-validation.00000-of-00008.json.gz` in `c4/en/validation`)

2. **Run** the `rqa-dataset-10M-1M.py` script to generate `.fvecs` files containing the base 
   (10M and 1M versions), ID and OOD query and learning set vectors. **Remember to set the path** to the folder where the 
   downloaded C4 files are located. 
   
> **_NOTE:_**  `rqa-dataset-10M-1M.py` makes several sequential calls to `generate_rqa_embeddings` 
> to generate the embeddings in batches. We recommend avoiding this and launching these calls in parallel to speed-up the process.
   

3. **Generate the ground-truth** by conducting an exhaustive search with the inner product metric. 
   We provide the [ground-truth](groundtruth) files for the query and learning sets (ID and OOD),
   for both the 10M and 1M versions of the base vectors sets.
   
> **_NOTE:_**  Due to floating-point arithmetic precision the vector embeddings generated using the provided
> code in different machines may slightly vary. Keep in mind that this could cause small discrepancies with the provided ground-truth.  

4. Functions `read_fvecs` and `read_ivecs` can be used to read `.fvecs` and `.ivecs` files respectively.

## References
Reference to cite when you use datasets generated with this code in a research paper:

```
@article{aguerrebere2023similarity,
        title={LeanVec: Search your vectors faster by making them fit.},
        journal = {arxiv},
        author={Mariano Tepper and Ishwar Bhati and Cecilia Aguerrebere and Mark Hildebrand and Ted Willke},        
        year = {2023}
}
```

<a id="1">[1]</a> 
Qu, Y.; Ding, Y.; Liu, J.; Liu, K.; Ren, R.; Zhao, W. X.; Dong, D.; Wu, H. and Wang, H..: RocketQA: 
An optimized training approach to dense passage retrieval for open-domain question answering. In:
Conference of the North American Chapter of the Association for Computational Linguistics: Human
Language Technologies. 5835–5847. (2021)

<a id="2">[2]</a> 
Raffel, C.; Shazeer, N.; Roberts, A.; Lee, K.; Narang, S.; Matena, M.; Zhou, Y.; Li, W.; Liu, 
P.J.: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. 
In: The Journal of Machine Learning Research 21,140:1–140:67.(2020)

<a id="3">[3]</a>
Tepper M.; Bhati I.; Aguerrebere, C.; Hildebrand M.; Willke T.: LeanVec: Search your vectors faster by making them fit. 
arXiv preprint arXiv:2312.16335 (2024)

This "research quality code"  is for Non-Commercial purposes provided by Intel "As Is" without any express or implied 
warranty of any kind. Please see the dataset's applicable license for terms and conditions. Intel does not own the 
rights to this data set and does not confer any rights to it. Intel does not warrant or assume responsibility for the accuracy or completeness of any information, text, graphics, links or other items within the code. A thorough security review has not been performed on this code. Additionally, this repository may contain components that are out of date or contain known security vulnerabilities.