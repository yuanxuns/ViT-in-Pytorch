import argparse
import os

import cv2
import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from src.dataset.mnist_color_texture_dataset import MnistDataset
from src.model.vit import ViT


def get_accuracy(model, mnist_loader, device):
    num_total = 0.0
    num_correct = 0.0

    for data in tqdm(mnist_loader):
        im = data["image"].float().to(device)
        number_cls = data["number_cls"].long().to(device)
        model_output = model(im)
        pred_num_cls_idx = torch.argmax(model_output, dim=-1)
        num_total += pred_num_cls_idx.size(0)
        num_correct += torch.sum(pred_num_cls_idx == number_cls).item()
    num_accuracy = num_correct / num_total
    print("Number Accuracy : {:2f}".format(num_accuracy))


def visualize_pos_embed(model):
    # pos_embed = 1 x Num_patches+1 x D
    # Get indexes after CLS
    pos_emb = model.patch_emb.pos_emb.detach().cpu()[0][1:]
    num_rows = model.patch_emb.image_height // model.patch_emb.patch_height
    num_cols = model.patch_emb.image_width // model.patch_emb.patch_width
    selected_rows = list(range(0, num_rows, 2))
    selected_cols = list(range(0, num_cols, 2))
    num_plots = len(selected_rows) * len(selected_cols)

    plt.tight_layout(pad=0.1, rect=(0.1, 0.1, 0.9, 0.9))
    fig, axs = plt.subplots(len(selected_rows), len(selected_cols), squeeze=False)
    count = 0
    for i in tqdm(range(num_rows * num_cols)):
        row = i // num_cols
        col = i % num_cols
        if row % 2 == 0 and col % 2 == 0:
            out = torch.cosine_similarity(pos_emb[i], pos_emb, dim=-1)
            fig.add_subplot(len(selected_rows), len(selected_cols), count + 1)
            plt.xticks([])
            plt.yticks([])
            count += 1
            plt.subplots_adjust(0.1, 0.1, 0.9, 0.9)
            plt.imshow(out.reshape(num_rows, num_cols), vmin=-1, vmax=1)
    for idx, ax in enumerate(axs.flat):
        if idx >= num_plots:
            ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
    os.makedirs("src/output", exist_ok=True)
    plt.savefig("src/output/position_plot.png", bbox_inches="tight")
    plt.close(fig)


def visualize_attn_weights(mnist, model, device):

    num_images = 10
    idxs = torch.randint(0, len(mnist) - 1, (num_images,))
    ims = torch.cat([mnist[idx]["image"][None, :] for idx in idxs]).float()
    ims = ims.to(device)
    attentions = []

    def get_attention(module, inputs, output):
        attentions.append(output.detach().cpu())

    # Add forward hook
    handles = []
    for name, module in model.named_modules():
        if "attn_dropout" in name:
            handles.append(module.register_forward_hook(get_attention))

    model(ims)
    for handle in handles:
        handle.remove()

    # Handle residuals
    attentions = [
        (torch.eye(att.size(-1), dtype=att.dtype) + att)
        / (torch.eye(att.size(-1), dtype=att.dtype) + att).sum(dim=-1).unsqueeze(-1)
        for att in attentions
    ]

    result = torch.max(attentions[0], dim=1)[0]
    # Max or mean both are fine
    for i in range(1, len(attentions)):
        att = torch.max(attentions[i], dim=1)[0]
        result = torch.matmul(att, result)

    masks = result
    masks = masks[:, 0, 1:]
    num_rows = model.patch_emb.image_height // model.patch_emb.patch_height
    num_cols = model.patch_emb.image_width // model.patch_emb.patch_width
    for i in range(num_images):
        im_input = torch.permute(ims[i].detach().cpu(), (1, 2, 0)).numpy()
        im_input = im_input[:, :, [2, 1, 0]]
        im_input = (im_input + 1) / 2 * 255
        mask = masks[i].reshape((num_rows, num_cols)).numpy()

        mask = mask / np.max(mask)

        mask = cv2.resize(
            mask,
            (model.patch_emb.image_width, model.patch_emb.image_height),
            interpolation=cv2.INTER_LINEAR,
        )[..., None]
        os.makedirs("src/output", exist_ok=True)
        cv2.imwrite("src/output/input_{}.png".format(i), im_input)
        cv2.imwrite("src/output/overlay_{}.png".format(i), im_input * mask)


def inference(args):

    with open(args.config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    requested_device = config["train_params"]["device"]
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    else:
        device = requested_device
    # Create the model and dataset
    model = ViT(config["model_params"]).to(device)
    mnist = MnistDataset(
        "test",
        config["dataset_params"],
        im_h=config["model_params"]["image_height"],
        im_w=config["model_params"]["image_width"],
    )
    mnist_loader = DataLoader(
        mnist,
        batch_size=config["train_params"]["batch_size"],
        shuffle=False,
        num_workers=4,
    )

    task_dir = os.path.join("src", config["train_params"]["task_name"])

    # Load checkpoint if found
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
    else:
        print(
            "No checkpoint found at {}".format(
                os.path.join(
                    task_dir,
                    config["train_params"]["ckpt_name"],
                )
            )
        )
    model.eval()
    with torch.no_grad():
        # Run inference and measure accuracy on number
        get_accuracy(model, mnist_loader, device)
        # Visualize positional embedding
        visualize_pos_embed(model)
        # Visualize attention weights
        visualize_attn_weights(mnist, model, device)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        default="src/config/config.yaml",
        help="Path to the YAML config file.",
    )
    return parser


if __name__ == "__main__":
    inference(build_parser().parse_args())
