import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

# Dependencies: Ensure the following classes are defined in this file or imported correctly
# from your_module import ResDWC, LayerNorm2d, StokenAttention, Mlp

class SlotAttention(nn.Module):
    """
    Slot Attention module from Locatello et al. (NeurIPS 2020).

    Args:
        num_slots: int, number of slots (K).
        dim: int, dimensionality of input features and slot vectors (E).
        iters: int, number of attention iterations (T).
        eps: float, numerical stability constant.
    """
    def __init__(self, num_slots: int, dim: int, iters: int = 3, eps: float = 1e-8):
        super(SlotAttention, self).__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters
        self.eps = eps

        # Parameters for slot initialization (learnable)
        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, dim))
        self.slots_sigma = nn.Parameter(torch.abs(torch.randn(1, num_slots, dim)))

        # Linear maps for computing queries, keys, and values
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        # GRU for slot update
        self.gru = nn.GRUCell(dim, dim)
        # MLP for slot refinement
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )
        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)

    def forward(self, inputs: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        inputs: torch.Tensor of shape (B, N, E)
        returns:
            slots: (B, K, E)
            attn: (B, N, K)
        """
        B, N, E = inputs.shape
        assert E == self.dim, "Input feature dimension must match slot dimension"

        # Normalize inputs
        x = self.norm_inputs(inputs)
        # Compute keys and values
        k = self.to_k(x)   # (B, N, E)
        v = self.to_v(x)   # (B, N, E)

        # Initialize slots from learned Gaussian parameters
        slots = self.slots_mu + torch.randn(B, self.num_slots, self.dim, device=inputs.device) * self.slots_sigma

        for _ in range(self.iters):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)
            q = self.to_q(slots_norm)  # (B, K, E)
            # Scaled dot-product attention
            q = q * (self.dim ** -0.5)

            # Compute attention logits: (B, N, K)
            logits = torch.einsum('bne,bke->bnk', k, q)
            # Attention weights over slots for each input
            attn = F.softmax(logits, dim=-1) + self.eps  # (B, N, K)
            # Normalize over inputs for each slot
            attn_norm = attn / torch.sum(attn, dim=1, keepdim=True)

            # Weighted mean: (B, K, E)
            updates = torch.einsum('bnk,bne->bke', attn_norm, v)

            # Slot update via GRU
            slots = self.gru(
                updates.view(-1, E),
                slots_prev.view(-1, E)
            )
            slots = slots.view(B, self.num_slots, E)

            # MLP refinement
            slots = slots + self.mlp(self.norm_mlp(slots))

        # Return final slots and last attention (inputs-to-slots)
        return slots, attn

# Example usage:
# B, N, E = 2, 64, 128
# x = torch.randn(B, N, E)\#
# slot_attn = SlotAttention(num_slots=7, dim=E)
# slots, attn = slot_attn(x)
# print(slots.shape)  # -> (2, 7, 128)
# print(attn.shape)   # -> (2, 64, 7)



class LearnableSampler(nn.Module):
    """
    Learnable sampler that aggregates an input token sequence of shape (B, N, E)
    into K representative tokens of shape (B, K, E) using cross-attention.

    If dynamic_q=True, queries are generated from the input x;
    otherwise, a learnable static query_embed is used.
    """
    def __init__(self, embed_dim, num_samples, num_heads=8, dynamic_q=False):
        super().__init__()
        self.num_samples = num_samples
        self.dynamic_q = dynamic_q
        if not self.dynamic_q:
            # Learnable static queries
            self.query_embed = nn.Parameter(torch.randn(1, num_samples, embed_dim))
        else:
            # Dynamic query generation: global pooling -> MLP to produce K * E values
            self.query_mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, num_samples * embed_dim)
            )
        # Native MultiheadAttention with batch_first=True
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, N, E)
        Returns:
            out: Tensor of shape (B, K, E)
        """
        B, N, E = x.shape
        # Build queries
        if self.dynamic_q:
            # Global average pooling to build context
            context = x.mean(dim=1)           # (B, E)
            # MLP generates (B, K * E)
            q = self.query_mlp(context)       # (B, K*E)
            q = q.view(B, self.num_samples, E)  # (B, K, E)
        else:
            # Expand static learnable queries to batch
            q = self.query_embed.expand(B, -1, -1)  # (B, K, E)
        # Cross-attention: query=q, key=x, value=x
        out, attn_weights = self.attn(q, x, x)
        return out

class StokenAttentionLayerWithSampler(nn.Module):
    """
    STViT Attention Layer extended with a learnable sampler,
    followed by normalization and MLP smoothing of the sampled tokens.
    Supports input of shape (B, C, H, W) or (B, N, E).
    Returns the spatial feature map and K sampled tokens.
    """
    def __init__(
        self,
        dim,
        n_iter,
        stoken_size,
        num_heads=1,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        act_layer=nn.GELU,
        layerscale=False,
        init_values=1e-5,
        sampler_k=64,
        sampler_heads=4,
        dynamic_q=False
    ):
        super().__init__()
        # Core STViT modules
        self.layerscale = layerscale
        self.pos_embed = ResDWC(dim, 3)  # Depthwise conv positional embedding
        self.norm1 = LayerNorm2d(dim)
        self.attn = StokenAttention(
            dim,
            stoken_size=stoken_size,
            n_iter=n_iter,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp2 = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            act_layer=act_layer,
            drop=drop
        )
        if layerscale:
            # LayerScale parameters for stability
            self.gamma_1 = nn.Parameter(init_values * torch.ones(1, dim, 1, 1), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(1, dim, 1, 1), requires_grad=True)
        # Learnable sampler without gating
        self.sampler = LearnableSampler(
            embed_dim=dim,
            num_samples=sampler_k,
            num_heads=sampler_heads,
            dynamic_q=dynamic_q
        )
        # Normalization and MLP smoothing for sampled tokens
        self.sampler_norm = nn.LayerNorm(dim)
        self.sampler_mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            act_layer(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor of shape (B, C, H, W) or (B, N, E)
        Returns:
            spatial_out: Tensor of shape (B, C, H, W)
            sampled_tokens: Tensor of shape (B, K, C)
        """
        # Detect if input is a sequence
        is_sequence = (x.dim() == 3)
        if is_sequence:
            B, N, E = x.shape
            # Assume N = H * W
            H = W = int(math.sqrt(N))
            assert H * W == N, "Sequence length must be a perfect square"
            # Reshape to spatial map (B, E, H, W)
            x = x.permute(0, 2, 1).reshape(B, E, H, W)
        B, C, H, W = x.shape
        # 1) STViT attention block and MLP (with optional LayerScale)
        x_spatial = self.pos_embed(x)
        if self.layerscale:
            x_spatial = x_spatial + self.drop_path(
                self.gamma_1 * self.attn(self.norm1(x_spatial))
            )
            x_spatial = x_spatial + self.drop_path(
                self.gamma_2 * self.mlp2(self.norm2(x_spatial))
            )
        else:
            x_spatial = x_spatial + self.drop_path(self.attn(self.norm1(x_spatial)))
            x_spatial = x_spatial + self.drop_path(self.mlp2(self.norm2(x_spatial)))
        # 2) Flatten to token sequence (B, N, C)
        feat_flat = x_spatial.reshape(B, C, H * W).permute(0, 2, 1)
        # 3) Sample K tokens
        sampled_tokens = self.sampler(feat_flat)
        # 4) Normalize and smooth sampled tokens, then residual connect
        smooth = self.sampler_mlp(self.sampler_norm(sampled_tokens))
        sampled_tokens = sampled_tokens + self.drop_path(smooth)
        # 5) Return results in original format
        if is_sequence:
            # Return sequence form (B, N, C)
            return feat_flat, sampled_tokens
        # Return spatial map (B, C, H, W)
        return x_spatial, sampled_tokens


class SlotSampler(nn.Module):
    """
    Sampler using SlotAttention to aggregate an input token sequence of shape (B, N, E)
    into K representative tokens of shape (B, K, E).
    """
    def __init__(self, embed_dim: int, num_samples: int, iters: int = 3, eps: float = 1e-8):
        super().__init__()
        # Replace cross-attention with slot attention
        self.slot_attn = SlotAttention(num_slots=num_samples, dim=embed_dim, iters=iters, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, N, E)
        Returns:
            slots: Tensor of shape (B, K, E)
        """
        slots, _ = self.slot_attn(x)
        return slots



# Debug examples
if __name__ == "__main__":
    B, N, E = 2, 256, 128
    dummy_seq = torch.randn(B, N, E)
    layer = StokenAttentionLayerWithSampler(
        dim=E,
        n_iter=1,
        stoken_size=(4, 4),
        sampler_k=32,
        dynamic_q=True
    )
    seq_out, tok_out = layer(dummy_seq)
    print("seq_out shape:", seq_out.shape, "tok_out shape:", tok_out.shape)
    dummy_map = torch.randn(B, E, 16, 16)
    spat_out, tok_out2 = layer(dummy_map)
    print("spat_out shape:", spat_out.shape, "tok_out2 shape:", tok_out2.shape)
