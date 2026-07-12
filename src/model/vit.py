import torch.nn as nn
from src.model.patch_embed import PatchEmbedding

from src.model.attention import Attention


class TransformerLayer(nn.Module):
    def __init__(self, config):
        """Initialize a transformer encoder layer for Vision Transformer blocks.

        Args:
            config (dict): Model configuration containing:
                - ``emb_dim``: Embedding dimension of the token sequence.
                - ``drop_prob``: Dropout probability used in attention layers.
                - ``hidden_dim``: Hidden dimension of the feed-forward network.
                - ``ff_drop_prob``: Dropout probability used in the feed-forward block.
                - Attention-related fields required by ``Attention``.
        """
        super().__init__()
        self.config = config
        self.emb_dim = config["emb_dim"]
        self.drop_prob = config["drop_prob"]
        self.hidden_dim = config["hidden_dim"]
        self.ff_drop_prob = config["ff_drop_prob"]

        self.attn_norm = nn.LayerNorm(self.emb_dim)
        self.attn_block = Attention(config)
        self.ff_norm = nn.LayerNorm(self.emb_dim)

        self.ff_block = nn.Sequential(
            nn.Linear(self.emb_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.ff_drop_prob),
            nn.Linear(self.hidden_dim, self.emb_dim),
            nn.Dropout(self.ff_drop_prob),
        )

    def forward(self, x):
        """Apply attention and feed-forward updates with residual connections.

        Args:
            x (torch.Tensor): Input tensor of shape ``(B, N, D)``, where ``B``
                is batch size, ``N`` is the number of tokens, and ``D`` is
                ``emb_dim``.

        Returns:
            torch.Tensor: Output tensor of shape ``(B, N, D)`` after layer
                normalization, self-attention, and feed-forward refinement.
        """
        out = x + self.attn_block(self.attn_norm(x))
        out = out + self.ff_block(self.ff_norm(out))
        return out


class ViT(nn.Module):
    def __init__(self, config):
        """Initialize a Vision Transformer classifier.

        Args:
            config (dict): Model configuration containing:
                - ``emb_dim``: Embedding dimension of patch and transformer tokens.
                - ``n_layers``: Number of transformer encoder layers.
                - ``num_classes``: Number of target classes for classification.
                - Patch embedding and attention fields required by submodules.
        """
        super().__init__()
        self.emb_dim = config["emb_dim"]
        self.n_layers = config["n_layers"]
        self.num_classes = config["num_classes"]

        self.patch_emb = PatchEmbedding(config)

        self.layers = nn.ModuleList(
            [TransformerLayer(config) for _ in range(self.n_layers)]
        )

        self.norm = nn.LayerNorm(self.emb_dim)
        self.mlp = nn.Linear(self.emb_dim, self.num_classes)

    def forward(self, x):
        """Run image classification with patch embeddings and transformer layers.

        Args:
            x (torch.Tensor): Input image tensor of shape ``(B, C, H, W)``,
                where ``B`` is batch size, ``C`` is the number of channels,
                ``H`` is image height, and ``W`` is image width.

        Returns:
            torch.Tensor: Class logits of shape ``(B, num_classes)`` produced
                from the final class-token representation.

        Dim explanation:
            Images are first converted into a token sequence of shape
            ``(B, num_patches + 1, D)``, where ``num_patches`` is the number of image patches and
            ``+1`` accounts for the class token. After the transformer stack,
            only the class token at index ``0`` is passed to the classifier.
        """
        # (B, C, H, W) -> (B, num_patches + 1, D)
        x = self.patch_emb(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        # (B, num_patches + 1, D) -> (B, num_classes)
        return self.mlp(x)[:, 0]
