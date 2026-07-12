import torch
import torch.nn as nn
from einops import rearrange, repeat


class PatchEmbedding(nn.Module):
    def __init__(self, config):
        """Initialize a patch embedding layer for Vision Transformer inputs.

        Args:
            config (dict): Model configuration containing:
                - ``patch_height``: Height of each image patch.
                - ``patch_width``: Width of each image patch.
                - ``image_height``: Input image height.
                - ``image_width``: Input image width.
                - ``im_channels``: Number of input image channels.
                - ``emb_dim``: Output embedding dimension for each patch token.
                - ``drop_prob``: Dropout probability applied after adding position embeddings.
        """
        super().__init__()
        self.config = config
        self.patch_height = config["patch_height"]
        self.patch_width = config["patch_width"]
        self.image_height = config["image_height"]
        self.image_width = config["image_width"]
        self.im_channels = config["im_channels"]
        self.emb_dim = config["emb_dim"]
        self.drop_prob = config["drop_prob"]

        num_patches = (self.image_height // self.patch_height) * (
            self.image_width // self.patch_width
        )

        patch_dim = self.im_channels * self.patch_height * self.patch_width

        self.patch_emb = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
        )

        self.pos_emb = nn.Parameter(torch.zeros(1, num_patches + 1, self.emb_dim))
        self.cls_token = nn.Parameter(torch.randn(self.emb_dim))
        self.patch_emb_dropout = nn.Dropout(self.drop_prob)

    def forward(self, x):
        """Convert images into patch tokens with class and positional embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape
                ``(B, C, H, W)``, where ``B`` is batch size, ``C`` is the
                number of channels, ``H`` is image height, and ``W`` is image width.

        Returns:
            torch.Tensor: Patch embedding tensor of shape
                ``(B, num_patches + 1, D)``
                ``+1`` is for the class token, and ``D`` is ``emb_dim``.

        Dim explanation:
            The input image is split into non-overlapping patches of size
            ``(patch_height, patch_width)``. Each patch is flattened from
            ``(C, patch_height, patch_width)`` to a vector of size
            ``patch_dim = C * patch_height * patch_width``, then projected
            to the embedding dimension ``D``.
        """
        # (B, im_channels, image_height, image_width) -> (B, num_patches, patch_dim)
        out = rearrange(
            x,
            "b c (nh ph) (nw pw) -> b (nh nw) (ph pw c)",
            ph=self.patch_height,
            pw=self.patch_width,
        )

        # (B, num_patches, patch_dim) - > (B, num_patches, emb_dim)
        out = self.patch_emb(out)

        # prepend class token
        batch_size = out.shape[0]

        # (emb_dim) -> (B, 1, emb_dim)
        cls_tokens = repeat(self.cls_token, "d -> b 1 d", b=batch_size)

        # (B, 1, emb_dim) + (B, num_patches, emb_dim) -> (B, num_patches + 1, emb_dim)
        out = torch.cat([cls_tokens, out], dim=1)

        # (B, num_patches + 1, emb_dim) + (1, num_patches + 1, emb_dim) -> (B, num_patches + 1, emb_dim)
        out = out + self.pos_emb
        out = self.patch_emb_dropout(out)

        return out
