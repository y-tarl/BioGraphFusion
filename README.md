# BioGraphFusion

The code for the paper ["BioGraphFusion: Graph Knowledge Embedding for Biological Completion and Reasoning"]

## Introduction

![Overall Diagram](./overall.png)

## Dependencies

- torch == 1.12.1
- torch_scatter == 2.0.9
- numpy == 1.21.6
- scipy == 1.10.1

### Reproduction with training scripts

##### Disease-Gene Prediction

```bash
python3 train.py --data_path ./data/Disease-Gene/DisGeNet_cv --train --topk 800 --layers 6 --fact_ratio 0.92 --gpu 0
```

##### Protein-Chemical Prediction

```bash
python3 train.py --data_path ./data/Protein-Chemical/STITCH --train --topk 300 --layers 6 --fact_ratio 0.92 --gpu 0
```

##### UMLS dataset

```bash
python3 train.py --data_path ./data/umls/ --train --topk 100 --layers 5 --fact_ratio 0.90 --gpu 0
```
