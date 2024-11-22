import math

import torch
from torch import nn


class LayerScale(nn.Module):
    """Learnable Layer Scale as proposed in https://arxiv.org/pdf/2103.17239"""

    def __init__(self, dims, input_size, scale_init=1e-5):
        super().__init__()
        scale_dim = (1,) * (dims - 1) + (input_size,)
        self.scale = nn.Parameter(torch.full(scale_dim, scale_init), requires_grad=True)

    def forward(self, x):
        return x * self.scale


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_p):
        super().__init__()
        self.d_model = d_model
        self.heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_head = d_model // num_heads

        self.attn_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x, pos_emb, mask=None):
        """
        Args:
            x: (B, T, F)
            pos_emb: (T/2^R, T/2^R, D) where D - head dim
        Returns:
            x: (B, T, F)
        """
        batch_size = x.size(0)

        q, k, v = self.attn_proj(x).split(dim=-1, split_size=self.d_model)
        q = q.view(batch_size, -1, self.heads, self.d_head).transpose(1, 2)  # (B, H, T, D)
        k = k.view(batch_size, -1, self.heads, self.d_head).transpose(1, 2)  # (B, H, T, D)
        v = v.view(batch_size, -1, self.heads, self.d_head).transpose(1, 2)  # (B, H, T, D)

        scores = torch.matmul(q, k.transpose(-2, -1))
        if pos_emb is not None:
            q_pos = q.contiguous().view(batch_size * self.heads, -1, self.d_head).transpose(0, 1)
            pos_scores = torch.matmul(q_pos, pos_emb.transpose(-2, -1))
            pos_scores = pos_scores.transpose(0, 1).view(batch_size, self.heads, pos_emb.size(0), pos_emb.size(1))
            scores = scores + pos_scores

        attn = torch.softmax(scores / math.sqrt(self.d_head), dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, -1, self.heads * self.d_head)
        out = self.out_proj(out)
        out = self.dropout(out)

        return out


class RelativePositionalEncoding(nn.Module):
    def __init__(self, emb_dim, maxlen):
        super().__init__()
        self.maxlen = maxlen
        self.pe_k = nn.Embedding(2 * maxlen, emb_dim)

    def forward(self, positions):
        positions = torch.clamp(positions, -self.maxlen, self.maxlen - 1) + self.maxlen
        pe_k = self.pe_k(positions)
        return pe_k
