# Vision Transformer for Colored-Texture MNIST

This repository contains a compact Vision Transformer (ViT) implementation in PyTorch for classifying MNIST digits rendered on textured backgrounds. The project focuses on the full modeling pipeline: patch embedding, multi-head self-attention, transformer encoder layers, training, inference, and simple attention visualizations.

## Overview

The model takes an RGB image, splits it into non-overlapping patches, embeds each patch into a token sequence, prepends a learnable class token, and processes the sequence with transformer encoder layers. The final class-token representation is passed to a linear classifier to predict the digit class.

The included dataset loader works with a custom MNIST-based dataset where:

- digit foregrounds are colorized
- backgrounds are textured
- training uses random crops of texture images
- evaluation uses centered crops

Although the dataset entries contain texture labels and digit color values, the current training loop optimizes only the digit classification objective.


## Model Architecture

### 1. Patch Embedding

Implemented in [`src/model/patch_emb.py`](/home/yuanxun/Code/vit/src/model/patch_emb.py).

- Splits an input image of shape `(B, C, H, W)` into patches
- Flattens each patch into a vector
- Projects each patch into `emb_dim`
- Prepends a learnable class token
- Adds learnable positional embeddings

For the default configuration:

- image size: `224 x 224`
- patch size: `16 x 16`
- number of patches: `14 x 14 = 196`
- token sequence length: `197` including the class token

### 2. Multi-Head Self-Attention

Implemented in [`src/model/attention.py`](/home/yuanxun/Code/vit/src/model/attention.py).

- projects input tokens into queries, keys, and values
- reshapes them into multiple attention heads
- computes scaled dot-product attention
- concatenates head outputs
- projects back to the embedding dimension

### 3. Transformer Layer

Implemented in [`src/model/vit.py`](/home/yuanxun/Code/vit/src/model/vit.py).

Each encoder block contains:

- LayerNorm before attention
- multi-head self-attention
- residual connection
- LayerNorm before feed-forward network
- MLP with GELU and dropout
- residual connection

### 4. ViT Classifier

Also implemented in [`src/model/vit.py`](/home/yuanxun/Code/vit/src/model/vit.py).

- stacks `n_layers` transformer blocks
- applies a final LayerNorm
- uses the class token for digit classification

## Dataset preparation
For setting up the mnist dataset: Follow - https://github.com/explainingai-code/Pytorch-VAE#data-preparation

Download Quarter RGB resolution texture data from ALOT Homepage In case you want to train on higher resolution, you can download that as well and but you would have to create new imdb.json Rest of the code should work fine as long as you create valid json files.

Download imdb.json from Drive Verify the data directory has the following structure after textures download

src/data/textures/{texture_number}
	*.png
src/data/train/images/{0/1/.../9}
	*.png
src/data/test/images/{0/1/.../9}
	*.png
src/data/imdb.json

## Configuration

The main configuration lives in [`src/config/config.yaml`](/home/yuanxun/Code/vit/src/config/config.yaml).


## Installation

This repo does not currently include a `requirements.txt` or `pyproject.toml`, so dependencies need to be installed manually.

Recommended environment:

- Python 3.10+
- PyTorch

Example installation:

```bash
pip install torch numpy opencv-python matplotlib pyyaml tqdm einops tensorboard
```

If you want CUDA support, install the appropriate PyTorch build from the official PyTorch installation guide.

## Training


```bash
python -m src.train
```

### TensorBoard

```bash
tensorboard --logdir src/ViT/tensorboard
```

If you change `task_name` in the config, update the log directory accordingly.

## Inference and Visualization

```bash
python -m src.inference
```

