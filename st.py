"""
StokenAttention + Soft‑Top‑K Sampler (after n_iter refinement)
=============================================================
This is a **drop‑in file** that adds a *differentiable* Top‑K selector to your
original StokenAttention pipeline while keeping the Unfold/Fold + multi‑iter
logic intact.

Key knobs
---------
* ``sample_topk``  – number of representative pixels to keep **per stoken‑patch**
* ``soft_sampler`` – ``True`` (default) uses the sparsemax‑style *soft* Top‑K
  projection; ``False`` falls back to the classic hard ``torch.topk`` gather.

Behaviour matrix
----------------
| sample_topk | soft_sampler | Return shape | Notes |
|-------------|--------------|--------------|-------|
| ``None``    |  any         | ``(B,C,H,W)`` | Original dense output |
| ``k``       | ``False``    | ``(B,K,C)``   | Hard Top‑K, no gradients through indices |
| ``k``       | ``True``     | ``(B,K,C)``   | *Soft* Top‑K, fully differentiable |

A small ``TokensToMap`` adapter is included for reshaping a CLIP/Vision‑Transformer
sequence ``(B, N, E)`` back into a spatial tensor ``(B, E, H', W')`` before the
sampler.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

# -----------------------------------------------------------------------------
# Adapter: ViT/CLIP tokens → feature map
# -----------------------------------------------------------------------------
class TokensToMap(nn.Module):
    """Convert a ViT/CLIP token sequence (B, N, E) to (B, E, H', W')."""
    def __init__(self, patch_size: int, *, has_cls: bool = True,
                 hw: tuple[int, int] | None = None):
        super().__init__()
        self.has_cls = has_cls
        self.hw = hw

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,N,E)
        if self.has_cls:
            x = x[:, 1:, :]            # drop CLS
        B, N, E = x.shape
        if self.hw is None:
            H_ = W_ = int(math.isqrt(N))
            assert H_ * W_ == N, "Token count N is not square – supply hw=(H',W')"
        else:
            H_, W_ = self.hw
            assert H_ * W_ == N, "Provided hw doesn't match sequence length"
        return x.reshape(B, H_, W_, E).permute(0, 3, 1, 2).contiguous()  # B,E,H',W'

# -----------------------------------------------------------------------------
# Helper layers
# -----------------------------------------------------------------------------
class LayerNorm2d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class ResDWC(nn.Module):
    def __init__(self, dim: int, k: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim)
    def forward(self, x):
        return x + self.conv(x)

class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio: float = 4., drop: float = 0.):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, dim, 1)
        self.drop = nn.Dropout(drop)
        self.conv = ResDWC(hidden, 3)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.conv(x)
        x = self.fc2(x); x = self.drop(x)
        return x

# -----------------------------------------------------------------------------
# Unfold / Fold helpers (depth‑wise im2col and col2im)
# -----------------------------------------------------------------------------
class Unfold(nn.Module):
    def __init__(self, k: int = 3):
        super().__init__()
        w = torch.eye(k * k).reshape(k * k, 1, k, k)
        self.register_buffer("w", w, persistent=False)
        self.k = k
    def forward(self, x):
        b, c, h, w = x.shape
        x = F.conv2d(x.reshape(b * c, 1, h, w), self.w, padding=self.k // 2)
        return x.reshape(b, c * (self.k ** 2), h * w)

class Fold(nn.Module):
    def __init__(self, k: int = 3):
        super().__init__()
        w = torch.eye(k * k).reshape(k * k, 1, k, k)
        self.register_buffer("w", w, persistent=False)
        self.k = k
    def forward(self, x):
        return F.conv_transpose2d(x, self.w, padding=self.k // 2)

# -----------------------------------------------------------------------------
# Single‑head spatial Attention (lightweight)
# -----------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, dim, num_heads: int = 8, qkv_bias=False, qk_scale=None,
                 attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(x).reshape(B, self.num_heads, C // self.num_heads * 3, H * W).chunk(3, dim=2)
        attn = (k.transpose(-1, -2) @ q) * self.scale
        attn = self.attn_drop(attn.softmax(-2))
        x = (v @ attn).reshape(B, C, H, W)
        return self.proj_drop(self.proj(x))

# -----------------------------------------------------------------------------
# Soft‑Top‑K projection (sparsemax‑style)
# -----------------------------------------------------------------------------

def soft_topk(scores: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
    """Return *k‑sparse* probs along `dim`. Fully differentiable."""
    topk_vals, _ = scores.topk(k, dim=dim)
    tau = (topk_vals.sum(dim=dim, keepdim=True) - 1) / k
    probs = (scores - tau).clamp(min=0.0)
    probs = probs / (probs.sum(dim=dim, keepdim=True) + 1e-9)
    return probs

# -----------------------------------------------------------------------------
# StokenAttention + optional Top‑K Sampler
# -----------------------------------------------------------------------------
class StokenAttention(nn.Module):
    def __init__(self, dim: int, stoken_size: tuple[int, int], *,
                 n_iter: int = 1, sample_topk: int | None = None,
                 soft_sampler: bool = True):
        super().__init__()
        self.h_patch, self.w_patch = stoken_size
        self.n_iter = n_iter
        self.sample_topk = sample_topk
        self.soft_sampler = soft_sampler
        self.scale = dim ** -0.5
        self.unfold = Unfold(3); self.fold = Fold(3)

    # ----------------------------- core ----------------------------------
    def forward(self, x: torch.Tensor):
        """x: (B,C,H,W) → dense map or (B,K,C) tokens"""
        B, C, H0, W0 = x.shape
        hp, wp = self.h_patch, self.w_patch
        pad_r = (wp - W0 % wp) % wp; pad_b = (hp - H0 % hp) % hp
        if pad_r or pad_b:
            x = F.pad(x, (0, pad_r, 0, pad_b))
        _, _, H, W = x.shape
        hh, ww = H // hp, W // wp   # number of patches
        M, L = hh * ww, hp * wp

        # 1) initial super‑tokens (avg pool)
        stoken = F.adaptive_avg_pool2d(x, (hh, ww))  # (B,C,hh,ww)
        pixels = x.view(B, C, hh, hp, ww, wp).permute(0, 2, 4, 3, 5, 1).reshape(B, M, L, C)  # (B,M,L,C)

        # 2) n_iter refinement
        for itr in range(self.n_iter):
            st_unf = self.unfold(stoken).transpose(1, 2).reshape(B, M, C, 9)  # (B,M,C,9)
            affinity = (pixels @ st_unf) * self.scale                         # (B,M,L,9)
            affinity = affinity.softmax(-1)
            if itr < self.n_iter - 1:
                stoken = pixels.transpose(-1, -2) @ affinity                  # (B,M,C,9)
                stoken = self.fold(stoken.permute(0, 2, 3, 1).reshape(B * C, 9, hh, ww)).reshape(B, C, hh, ww)
                norm = self.fold(affinity.sum(2).transpose(1, 2).reshape(B, 9, hh, ww)) + 1e-12
                stoken = stoken / norm

        # 3) reconstruct full pixel map (needed for hard gather or dense out)
        st_unf = self.unfold(stoken).transpose(1, 2).reshape(B, M, C, 9)
        pixels_full = st_unf @ affinity.transpose(-1, -2)                      # (B,M,C,L)
        pixels_full = pixels_full.reshape(B, hh, ww, C, hp, wp)
        pixels_full = pixels_full.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

        # ------------------- Top‑K sampler ---------------------
        if self.sample_topk is not None:
            k = min(self.sample_topk, L)  # safety
            pixel_seq = pixels_full.view(B, C, hh, hp, ww, wp).permute(0, 2, 4, 3, 5, 1).reshape(B, M, L, C)
            score = affinity.mean(-1)                                       # (B,M,L)

            if self.soft_sampler:
                probs = soft_topk(score, k, dim=-1)                         # (B,M,L)
                sampled = (probs.unsqueeze(-1) * pixel_seq).sum(-2)         # (B,M,C)
                sampled = sampled.unsqueeze(-2).repeat(1, 1, k, 1)          # broadcast to k positions
            else:
                _, idx = score.topk(k, dim=-1)                              # (B,M,k)
                gather_idx = idx.unsqueeze(-1).expand(-1, -1, -1, C)
                sampled = torch.gather(pixel_seq, 2, gather_idx)            # (B,M,k,C)

            sampled = sampled.reshape(B, M * k, C)                          # (B,K,C)
            return sampled

        # dense fallback
        if pad_r or pad_b:
            pixels_full = pixels_full[..., :H0, :W0]
        return pixels_full

# -----------------------------------------------------------------------------
# Wrapper block to slot into a network
# -----------------------------------------------------------------------------
class StokenAttentionLayer(nn.Module):
    def __init__(self, dim: int, stoken_size: tuple[int, int], *,
                 n_iter: int = 1, mlp_ratio: float = 4., drop: float = 0.,
                 drop_path: float = 0., sample_topk: int | None = None,
                 soft_sampler: bool = True):
        super().__init__()
        self.pos = ResDWC(dim, 3)
        self.norm1 = LayerNorm2d(dim)
        self.attn = StokenAttention(dim, stoken_size, n_iter=n_iter,
                                    sample_topk=sample_topk, soft_sampler=soft_sampler)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = Mlp(dim, mlp_ratio, drop=drop)

    def forward(self, x):  # x: (B,C,H,W)
        x = self.pos(x)
        y = self.attn(self.norm1(x))
        if y.ndim == 4:  # dense path
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        # sequence path
        y = y + self.drop_path(y)
        y = self.mlp(self.norm2(y.transpose(1, 2).unsqueeze(-1))).squeeze(-1).transpose(1, 2)
        return y
