import argparse
import os
import random

import numpy as np
import torch
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset.mnist_color_texture_dataset import MnistDataset
from src.model.vit import ViT


def train_for_one_epoch(
    epoch_idx, model, mnist_loader, optimizer, device, writer, global_step
):
    # Accumulate batch losses so we can report a stable epoch-level mean.
    losses = []
    criterion = torch.nn.CrossEntropyLoss()
    progress_bar = tqdm(mnist_loader, desc=f"Epoch {epoch_idx + 1}")
    for data in progress_bar:
        im = data["image"].float().to(device)
        number_cls = data["number_cls"].long().to(device)

        # Standard forward/backward/update step for one mini-batch.
        optimizer.zero_grad()
        model_output = model(im)
        loss = criterion(model_output, number_cls)
        # Log batch loss immediately so TensorBoard shows fine-grained training dynamics.
        losses.append(loss.item())
        writer.add_scalar("train/loss_step", loss.item(), global_step)
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        global_step += 1
    print(
        "Finished epoch: {} | Number Loss : {:.4f}".format(
            epoch_idx + 1, np.mean(losses)
        )
    )
    # Return the mean loss and updated step counter to the outer training loop.
    return np.mean(losses), global_step


def train(args):
    # Load the experiment config that drives model, data, and optimization settings.
    with open(args.config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    # Seed all random number generators so runs are reproducible.
    seed = config["train_params"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    requested_device = config["train_params"]["device"]
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    else:
        device = requested_device

    if device == "cuda":
        # Seed all CUDA kernels as well when training on GPU.
        torch.cuda.manual_seed_all(seed)

    # Build the model and training dataset from the config.
    model = ViT(config["model_params"]).to(device)
    mnist = MnistDataset(
        "train",
        config["dataset_params"],
        im_h=config["model_params"]["image_height"],
        im_w=config["model_params"]["image_width"],
    )
    mnist_loader = DataLoader(
        mnist,
        batch_size=config["train_params"]["batch_size"],
        shuffle=True,
        num_workers=4,
    )
    num_epochs = config["train_params"]["epochs"]
    optimizer = Adam(model.parameters(), lr=config["train_params"]["lr"])
    # Reduce the learning rate when the epoch loss stops improving.
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

    # Create the run directory that will hold checkpoints and logs.
    task_dir = os.path.join("src", config["train_params"]["task_name"])
    os.makedirs(task_dir, exist_ok=True)

    log_dir = os.path.join(task_dir, "tensorboard")
    writer = SummaryWriter(log_dir=log_dir)

    # Resume from the most recent checkpoint when one is available.
    if os.path.exists(
        os.path.join(
            task_dir,
            config["train_params"]["ckpt_name"],
        )
    ):
        print("Loading checkpoint")
        model.load_state_dict(
            torch.load(
                os.path.join(
                    task_dir,
                    config["train_params"]["ckpt_name"],
                ),
                map_location=device,
            )
        )
    best_loss = np.inf
    global_step = 0

    for epoch_idx in range(num_epochs):
        mean_loss, global_step = train_for_one_epoch(
            epoch_idx, model, mnist_loader, optimizer, device, writer, global_step
        )
        writer.add_scalar("train/loss_epoch", mean_loss, epoch_idx)
        # Drive the scheduler from epoch-average loss rather than noisy batch loss.
        scheduler.step(mean_loss)
        # Keep only the best checkpoint seen so far.
        if mean_loss < best_loss:
            print("Improved Loss to {:.4f} .... Saving Model".format(mean_loss))
            torch.save(
                model.state_dict(),
                    os.path.join(
                    task_dir,
                    config["train_params"]["ckpt_name"],
                ),
            )
            best_loss = mean_loss
        else:
            print("No Loss Improvement")

    writer.close()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        default="src/config/config.yaml",
        help="Path to the YAML config file.",
    )
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
