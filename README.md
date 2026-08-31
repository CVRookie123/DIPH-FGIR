# Discriminative Information-Preserving Hashing for Fine-Grained Image Retrieval

Official PyTorch implementation of the paper **“Discriminative Information-Preserving Hashing for Fine-Grained Image Retrieval”**.

## Introduction

This repository provides the official implementation of **DIPH**, a fine-grained hashing method designed to improve the discriminability and semantic consistency of hash codes for fine-grained image retrieval.

## Requirements

Please install the required third-party libraries according to `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Datasets

We use the following public fine-grained image datasets:

* CUB-200-2011
* FGVC-Aircraft
* Food101
* NABirds
* VegFru

Please download the datasets from their **official public websites** and set the dataset root path using the `--root` argument when training.

## Training

Run the following commands to train DIPH on different datasets.

### CUB-200-2011

```bash
python main.py --dataset cub-2011 --root /path/to/CUB_200_2011 --max-epoch 30 --gpu 0 --batch-size 32 --max-iter 40 --code-length 12,24,32,48 --lr 5.0e-4 --wd 1e-4 --optim SGD --lr-step 40 --num-samples 2000 --num_classes 200 --momen=0.91
```

### Aircraft

```bash
python main.py --dataset aircraft --root /path/to/Aircraft --max-epoch 30 --gpu 0 --batch-size 32 --max-iter 40 --code-length 12,24,32,48 --lr 5.0e-4 --wd 1e-4 --optim SGD --lr-step 40 --num-samples 2000 --num_classes 100 --momen=0.91
```

### Food101

```bash
python main.py --dataset food101 --root /path/to/food-101 --max-epoch 30 --gpu 0 --batch-size 32 --max-iter 50 --code-length 12,24,32,48 --lr 5.0e-4 --wd 1e-4 --optim SGD --lr-step 40 --num-samples 2000 --num_classes 101 --momen 0.91
```

### NABirds

```bash
python main.py --dataset nabirds --root /path/to/nabirds --max-epoch 30 --gpu 0 --batch-size 32 --max-iter 50 --code-length 12,24,32,48 --lr 5.0e-4 --wd 1e-4 --optim SGD --lr-step 40 --num-samples 4000 --num_classes 555 --momen=0.91
```

### VegFru

```bash
python main.py --dataset vegfru --root /path/to/vegfru --max-epoch 30 --gpu 0 --batch-size 32 --max-iter 50 --code-length 12,24,32,48 --lr 5.0e-4 --wd 1e-4 --optim SGD --lr-step 40 --num-samples 4000 --num_classes 292 --momen=0.91
```

## Notes

* Please make sure the dataset path specified by --root is correct.

* GPU id can be specified through --gpu.

* Multiple hash code lengths can be trained/evaluated via --code-length 12,24,32,48.

* You may adjust training hyperparameters according to your hardware resources.