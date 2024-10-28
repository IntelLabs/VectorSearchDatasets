# Text Dataset Generator

This provides code to generate vector embedding datasets from text and save them to 
`fvecs` format for use in similarity search applications (e.g. retrieval-augmented 
generation with LLMs). The code assumes that the text embedding model is available 
through the popular [SentenceTransformers](https://sbert.net/) library.

Since many text datasets and embedding models are increasing in size, the code enables data 
parallelism in two different ways: by sharding the dataset samples (allowing multiple 
nodes to process different shards and combine later) and by running inference on 
multiple devices at once (allowing many GPUs within a node to be used on different 
parts of the data).

More specifically, we provide two examples to generate datasets using text 
from the [KILT](https://ai.meta.com/tools/kilt/) (Knowledge Intensive Language Tasks) 
benchmark Wikipedia dump [[1]](#1). Our examples use multi-GPU parallelism within a node, and 
process shards sequentially within that node. Here the sharding enables the user to 
process shards on a node with limited storage, and moves each shard to a final location 
with more storage.

### 3rd Party Dataset Disclaimer
Please see the dataset's applicable license for terms and conditions. Intel 
does not own the rights to this data set and does not confer any rights to 
it. Intel does not warrant or assume responsibility for the accuracy or 
completeness of any information, text, graphics, links or other items within 
the dataset.

## wiki-45M dataset

This version of the dataset relies on the original KILT repository archived 
[here](https://github.com/facebookresearch/KILT). To reproduce the dataset, one should clone
 this repository and follow the pip installation instructions. The script we have provided,
`preprocess_kilt_data.py`, is a modified version of `KILT/scripts/create_kilt_data_paragraphs.py`. 
The main differences are:
* Our version creates text chunks based on the number of words, rather than the number of 
`nltk` tokens. This means that punctuations are not counted separately in the chunk length, 
resulting in an overall reduction of the number of chunks (45M compared to 110M below).
* We found that the original preprocessing could produce confusing results if the vectors are 
used for text retrieval. To help alleviate this, we added a step to prepend the section title 
to a chunk if it begins with a bullet point.
* We modified the multiprocessing to rely on `tqdm.contrib.concurrent.process_map`, which 
automatically handles some functions like launching the multiprocessing pool.

To actually download the Wikipedia dataset and launch the preprocessing, follow the steps as 
outlined in the original repository. We summarize the commands below:
1. Download the 35GB KILT knowledge source as provided in the archived repository. Use mongoDB 
to index the knowledge base as follows:
```
wget http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json
mongoimport --db kilt --collection knowledgesource --file kilt_knowledgesource.json
```
2. Launch preprocessing, which splits the DB into however many `threads` you specify and saves them.
```
python preprocess_kilt_data.py --step preprocess --folder "./kilt_data" --threads 64
```
3. Divide the text data into 100 word chunks. To process one part of the database, provide that index 
as `rank`. This allows you to parallelize this step.
```
python preprocess_kilt_data.py --step main --chunk_size 100 --folder "./kilt_data" --rank <int>
```
4. Merge all of the parts of the preprocessed data together.
```
python preprocess_kilt_data.py --step merge --folder "./kilt_data" --threads 64
```
This will give us one `.jsonl` file that contains all of the text chunks for the entire dataset.

Now we can execute the `wikipedia_dataset.py` script with our desired parameters. We will pass the
preprocessed file to the `json_name` argument of the script. The example command below will use the
large, 1536-dimensional text embedding model created by Alibaba,
[gte-Qwen2-1.5B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct).

```
python wikipedia_dataset.py \
    --data_dir "./wiki_data_dir" --output_dir "./wiki_data_dir" \
    --embed_model "Alibaba-NLP/gte-Qwen2-1-5B-instruct" --embed_size 1536 \
    --dataset_name wiki
    --json_name "./kilt_data/kilt.jsonl" 
    --number_vec_shards 12 \
    --multiproc \
    --batch_size 64 --chunk_size 960 \
    --output_prefix wiki-45M --final_output_dir "./vector_data" 
```
The batch size and chunk size should be customized to your hardware. This will be limited by the GPU
memory you have available. All embeddings are written to memory-mapped numpy arrays.

If your job stops for any reason, you may resume the embedding by passing the index of the vector
where you want to continue processing the data with `--start_vector_index {vector_index}`. This index
should correspond to the number of total samples in the dataset, i.e. for wiki-45M that maximum is
45366081. The total number of samples will be printed out when you run the script.

The result of the script will be an `fvecs` file containing all of the embeddings. This will be created
from a numpy file that has merged all of the dataset shards together.


## wikipedia_110M dataset

You simply need to execute the `wikipedia_dataset.py` script with the desired 
arguments. We will be using an already preprocessed version of the KILT Wikipedia dataset available 
from the authors of the [RAGGED paper](https://github.com/neulab/ragged)[[2]](#2). 

In our example below, we'll be using a 1024-dimensional text embedding model created by BAAI, 
[bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5).
By dividing the dataset into 8 shards, we will create individual shards that are about ~50GB each.

Unlike the previous example, we provide an argument `--data_key`. This is because we use the HuggingFace 
`datasets` library to load the data, and it encodes the text data in one of the dataset fields. For
the preprocessed version we downloaded from the HuggingFace Hub, this field is `contents`.

```
python wikipedia_dataset.py \
    --data_dir "./wiki_data_dir" --output_dir "./wiki_data_dir" \
    --hf_path "jenhsia/ragged" --data_key "contents" \
    --dataset_name "kilt_wikipedia" \
    --embed_model "BAAI/bge-large-en-v1.5 --embed_size 1024 \
    --multiproc \
    --number_vec_shards 8 \
    --batch_size 512 --chunk_size 5000 \
    --output_prefix kilt_wikipedia_110M --final_output_dir "./vector_data" 
```

## References

<a id="1">[1]</a> 
Petroni, F.; Piktus, A.; Fan, A.; Lewis, P.; Yazdani, M.; De Cao, N.; Thorne, J.; Jernite, Y.; Karpukhin, V.; 
Maillard, J.; Plachouras, V.; Rockt{\"a}schel, T.; Riedel, S.: KILT: a Benchmark for Knowledge Intensive Language Tasks. 
In: Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: 
Human Language Technologies (NAACL). 2523-2544. (2021)

<a id="2">[2]</a>
Hsia, J.; Shaikh, A.; Wang, Z.; Neubig, G.: RAGGED: Towards Informed Design of Retrieval Augmented Generation Systems.
arXiv:2403.09040. (2024) 
