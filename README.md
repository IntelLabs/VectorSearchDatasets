# DPR Dataset Generator

This repository provides code to generate base and query vector datasets for similarity search benchmarking and evaluation on high-dimensional vectors stemming from large language models.
With the dense passage retriever (DPR) [[1]](#1), we encode text snippets from the C4 dataset [[2]](#2) to generate 768-dimensional vectors:
- context DPR embeddings for the base set and
- question DPR embeddings for the query set. 

The number of base and query embedding vectors is parametrizable. In [[3]](#3), 10 million base vectors and 10,000 query vectors are used.

## References

## References
Reference to cite when you use datasets generated with this code in a research paper:

```
@misc{aguerrebere2023similarity,
title={Similarity search in the blink of an eye with compressed indices},
author={Cecilia Aguerrebere and Ishwar Bhati and Mark Hildebrand and Mariano Tepper and Ted Willke},
year={2023},
eprint={2304.04759},
archivePrefix={arXiv},
primaryClass={cs.LG}
}
```

<a id="1">[1]</a> 
Karpukhin, V.; Oguz, B.; Min, S.; Lewis, P.; Wu, L.; Edunov, S.; Chen, D.; Yih, W..: Dense Passage 
Retrieval for Open-Domain Question Answering. In: Proceedings of the 2020 Conference on Empirical 
Methods in Natural Language Processing (EMNLP). 6769–6781. (2020)

<a id="2">[2]</a> 
Raffel, C.; Shazeer, N.; Roberts, A.; Lee, K.; Narang, S.; Matena, M.; Zhou, Y.; Li, W.; Liu, 
P.J.: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. 
In: The Journal of Machine Learning Research 21,140:1–140:67.(2020)

<a id="3">[3]</a>
Aguerrebere, C.; Bhati I.; Hildebrand M.; Tepper M.; Willke T.:Similarity search in the blink of an eye with compressed
indices. In: arXiv preprint [arXiv:2304.04759](https://arxiv.org/abs/2304.04759) (2023)
