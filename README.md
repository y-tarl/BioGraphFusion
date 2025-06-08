# BioGraphFusion

The code for the paper ["BioGraphFusion: Graph Knowledge Embedding for Biological Completion and Reasoning"]

## Introduction

BioGraphFusion is a framework that combines Knowledge Embeddings (KE) and Graph Structure Propagation (GSP) to improve biomedical knowledge graph reasoning and completion. It constructs a knowledge graph from fact triples, such as Drug-Disease and Protein-Chemical relationships, and employs Global Biological Tensor Encoding to learn latent biological associations. This process uses Canonical Polyadic (CP) decomposition to create embedding matrices for entities and relations.

The model then uses Query-Guided Subgraph Construction and Propagation to build a query-relevant subgraph, refining relationships and capturing context-specific semantics. The Scoring Integration module merges propagation results with globally-informed scores from KE, enhancing prediction accuracy and interpretability. By modeling both local interactions and global dependencies, BioGraphFusion advances biomedical graph reasoning. See the diagram below for an overview:

![Overall Diagram](./overall.png)

## Dependencies

- torch == 1.12.1
- torch_scatter == 2.0.9
- numpy == 1.21.6
- scipy == 1.10.1

### Reproduction with training scripts

##### Disease-Gene Prediction

```bash
python3 train.py --data_path ./data/Disease-Gene/DisGeNet_cv --topk 800 --layers 6 --fact_ratio 0.92 --gpu 0 
```

##### Protein-Chemical Prediction

```bash
python3 train.py --data_path ./data/Protein-Chemical/STITCH --topk 300 --layers 6 --fact_ratio 0.92 --gpu 0 
```

##### UMLS dataset

```bash
python3 train.py --data_path ./data/umls/ --topk 100 --layers 5 --fact_ratio 0.90 --gpu 0 
```

## Acknowledgements

This code is based on the work of [AdaProp](https://github.com/LARS-research/AdaProp)

