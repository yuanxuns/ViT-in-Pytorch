import argparse
import os
import random

import numpy as np
import torch
import yaml
from src.model.transformer import ViT
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset.mnist_color_texture_dataset import MnistDataset


def train_for_one_epoch(
    epoch_idx, model, mnist_loader, optimizer, device, writer, global_step
):
    losses = []
    criterion = torch.nn.CrossEntropyLoss()
    progress_bar = tqdm(mnist_loader, desc=f"Epoch {epoch_idx + 1}")
    for data in progress_bar:
        im = data["image"].float().to(device)
        number_cls = data["number_cls"].long().to(device)

        optimizer.zero_grad()
        model_output = model(im)
        loss = criterion(model_output, number_cls)
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
    return np.mean(losses), global_step


def train(args):
    #  Read the config file
    with open(args.config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    # Set the desired seed value
    seed = config["train_params"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = config["train_params"]["device"]
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Create the model and dataset
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
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

    # Create output directories
    if not os.path.exists("src/" + config["train_params"]["task_name"]):
        os.mkdir("src/" + config["train_params"]["task_name"])

    log_dir = os.path.join("src", config["train_params"]["task_name"], "tensorboard")
    writer = SummaryWriter(log_dir=log_dir)

    # Load checkpoint if found
    if os.path.exists(
        os.path.join(
            "src/" + config["train_params"]["task_name"],
            config["train_params"]["ckpt_name"],
        )
    ):
        print("Loading checkpoint")
        model.load_state_dict(
            torch.load(
                os.path.join(
                    "src/" + config["train_params"]["task_name"],
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
        scheduler.step(mean_loss)
        # Simply update checkpoint if found better version
        if mean_loss < best_loss:
            print("Improved Loss to {:.4f} .... Saving Model".format(mean_loss))
            torch.save(
                model.state_dict(),
                os.path.join(
                    "src/" + config["train_params"]["task_name"],
                    config["train_params"]["ckpt_name"],
                ),
            )
            best_loss = mean_loss
        else:
            print("No Loss Improvement")

    writer.close()
