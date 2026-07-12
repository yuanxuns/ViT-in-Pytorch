# Vision Transformer (ViT) in Pytorch

This repository contains a compact Vision Transformer (ViT) implementation in PyTorch for classifying MNIST digits rendered on textured backgrounds. The project focuses on the full modeling pipeline: patch embedding, multi-head self-attention, transformer encoder layers, training, inference, and simple attention visualizations.

## Overview

<img width="875" height="586" alt="Screenshot from 2026-07-11 20-05-03" src="https://github.com/user-attachments/assets/d38b3712-474a-4e0e-a11d-e9fc8746c961" />

The model takes an RGB image, splits it into non-overlapping patches, embeds each patch into a token sequence, prepends a learnable class token, and processes the sequence with transformer encoder layers. The final class-token representation is passed to a linear classifier to predict the digit class.

The included dataset loader works with a custom MNIST-based dataset where:

- digit foregrounds are colorized
- backgrounds are textured
- training uses random crops of texture images
- evaluation uses centered crops

Although the dataset entries contain texture labels and digit color values, the current training loop optimizes only the digit classification objective.

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

## Result

Attention Map Visualization

<img width="224" height="224" alt="input_9" src="https://github.com/user-attachments/assets/fbebe4f4-71c7-4972-ab7d-b01d9837f593" />
<img width="224" height="224" alt="overlay_9" src="https://github.com/user-attachments/assets/734e9440-ef16-4336-85c3-32e8810d8a8c" />

Positional Embedding Visualization

<img width="532" height="404" alt="position_plot" src="https://github.com/user-attachments/assets/6583c21a-2dd3-46d9-a5a4-0d66cd7e5f32" />

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


