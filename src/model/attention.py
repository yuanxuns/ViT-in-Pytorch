import torch
import torch.nn as nn
from einops import rearrange


class Attention(nn.Module):
    def __init__(self, config):
        """Initialize a multi-head self-attention block.

        Args:
            config (dict): Model configuration containing:
                - ``num_heads``: Number of attention heads.
                - ``head_dim``: Feature dimension of each attention head.
                - ``emb_dim``: Input and output embedding dimension.
                - ``drop_prob``: Dropout probability applied to attention weights
                  and the output projection.
        """
        super().__init__()
        self.config = config
        self.num_heads = config["num_heads"]
        self.head_dim = config["head_dim"]
        self.emb_dim = config["emb_dim"]
        self.drop_prob = config["drop_prob"]
        self.attn_dim = self.num_heads * self.head_dim

        self.qkv_proj = nn.Linear(self.emb_dim, 3 * self.attn_dim, bias=False)
        self.out_proj = nn.Sequential(
            nn.Linear(self.attn_dim, self.emb_dim), nn.Dropout(self.drop_prob)
        )
        self.attn_dropout = nn.Dropout(self.drop_prob)

    def forward(self, x):
        """Apply multi-head self-attention to a sequence of token embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape ``(B, N, D)``, where ``B``
                is batch size, ``N`` is the number of tokens, and ``D`` is
                ``emb_dim``.

        Returns:
            torch.Tensor: Output tensor of shape ``(B, N, D)`` after projecting
                the attended token representations back to the embedding space.

        Dim explanation:
            The input is projected into queries, keys, and values, then reshaped
            into ``num_heads`` attention heads of size ``head_dim``. Attention is
            computed independently per head and the results are concatenated back
            into the original embedding dimension.
        """
        B, N = x.shape[:2]
        qkv = self.qkv_proj(x)
        qkv = rearrange(
            qkv,
            "b n (three h d) -> three b h n d",
            three=3,
            h=self.num_heads,
            d=self.head_dim,
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = torch.einsum("bhid,bhjd->bhij", q, k) / (self.head_dim**0.5)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        out = torch.einsum("bhij,bhjd->bhid", attn_probs, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.out_proj(out)

        return out
