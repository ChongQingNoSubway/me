import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import scipy.io as sio
import torch.nn.functional as F
import math
from functools import partial
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import time

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()

class ResDWC(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        
        self.dim = dim
        self.kernel_size = kernel_size
        
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size//2, groups=dim)
                
        # self.conv_constant = nn.Parameter(torch.eye(kernel_size).reshape(dim, 1, kernel_size, kernel_size))
        # self.conv_constant.requires_grad = False
        
    def forward(self, x):
        # return F.conv2d(x, self.conv.weight+self.conv_constant, self.conv.bias, stride=1, padding=self.kernel_size//2, groups=self.dim) # equal to x + conv(x)
        return x + self.conv(x)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., conv_pos=True, downsample=False, kernel_size=5):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features = out_features or in_features
        self.hidden_features = hidden_features = hidden_features or in_features
               
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act1 = act_layer()         
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        
        self.conv = ResDWC(hidden_features, 3)
        
    def forward(self, x):       
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)        
        x = self.conv(x)        
        x = self.fc2(x)               
        x = self.drop(x)
        return x
 
class Attention(nn.Module):
    def __init__(self, dim, window_size=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
                
        self.window_size = window_size

        self.scale = qk_scale or head_dim ** -0.5
                
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)
        

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
                
        q, k, v = self.qkv(x).reshape(B, self.num_heads, C // self.num_heads *3, N).chunk(3, dim=2) # (B, num_heads, head_dim, N)
        
        attn = (k.transpose(-1, -2) @ q) * self.scale
        
        attn = attn.softmax(dim=-2) # (B, h, N, N)
        attn = self.attn_drop(attn)
        
        x = (v @ attn).reshape(B, C, H, W)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Unfold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        
        self.kernel_size = kernel_size
        
        weights = torch.eye(kernel_size**2)
        weights = weights.reshape(kernel_size**2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)
           
        
    def forward(self, x):
        b, c, h, w = x.shape
        x = F.conv2d(x.reshape(b*c, 1, h, w), self.weights, stride=1, padding=self.kernel_size//2)        
        return x.reshape(b, c*9, h*w)

class Fold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        
        self.kernel_size = kernel_size
        
        weights = torch.eye(kernel_size**2)
        weights = weights.reshape(kernel_size**2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)
           
        
    def forward(self, x):
        b, _, h, w = x.shape
        x = F.conv_transpose2d(x, self.weights, stride=1, padding=self.kernel_size//2)        
        return x

class StokenAttention(nn.Module):
    def __init__(self, dim, stoken_size, n_iter=1, refine=True, refine_attention=True, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        self.n_iter = n_iter
        self.stoken_size = stoken_size
        self.refine = refine
        self.refine_attention = refine_attention  
        
        self.scale = dim ** - 0.5
        
        self.unfold = Unfold(3)
        self.fold = Fold(3)
        
        if refine:
            
            if refine_attention:
                self.stoken_refine = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
            else:
                self.stoken_refine = nn.Sequential(
                    nn.Conv2d(dim, dim, 1, 1, 0),
                    nn.Conv2d(dim, dim, 5, 1, 2, groups=dim),
                    nn.Conv2d(dim, dim, 1, 1, 0)
                )
        
    def stoken_forward(self, x):
        '''
           x: (B, C, H, W)
        '''
        B, C, H0, W0 = x.shape
        h, w = self.stoken_size
        
        pad_l = pad_t = 0
        pad_r = (w - W0 % w) % w
        pad_b = (h - H0 % h) % h
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
            
        _, _, H, W = x.shape
        
        hh, ww = H//h, W//w
        
        # 976
        
        stoken_features = F.adaptive_avg_pool2d(x, (hh, ww)) # (B, C, hh, ww)
        # 955
        
        # 935
        pixel_features = x.reshape(B, C, hh, h, ww, w).permute(0, 2, 4, 3, 5, 1).reshape(B, hh*ww, h*w, C)
        # 911
        
        with torch.no_grad():
            for idx in range(self.n_iter):
                stoken_features = self.unfold(stoken_features) # (B, C*9, hh*ww)
                stoken_features = stoken_features.transpose(1, 2).reshape(B, hh*ww, C, 9)
                affinity_matrix = pixel_features @ stoken_features * self.scale # (B, hh*ww, h*w, 9)
                # 874
                affinity_matrix = affinity_matrix.softmax(-1) # (B, hh*ww, h*w, 9)
                # 871
                affinity_matrix_sum = affinity_matrix.sum(2).transpose(1, 2).reshape(B, 9, hh, ww)
                    # 777
                affinity_matrix_sum = self.fold(affinity_matrix_sum)
                if idx < self.n_iter - 1:
                    stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix # (B, hh*ww, C, 9)
                    # 853
                    stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B*C, 9, hh, ww)).reshape(B, C, hh, ww)            
                    # 777            
                    
                    # 771
                    stoken_features = stoken_features/(affinity_matrix_sum + 1e-12) # (B, C, hh, ww)
                    # 767
        
        stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix # (B, hh*ww, C, 9)
        # 853
        stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B*C, 9, hh, ww)).reshape(B, C, hh, ww)            
        
        stoken_features = stoken_features/(affinity_matrix_sum.detach() + 1e-12) # (B, C, hh, ww)
        # 767
        
        if self.refine:
            if self.refine_attention:
                # stoken_features = stoken_features.reshape(B, C, hh*ww).transpose(-1, -2)
                stoken_features = self.stoken_refine(stoken_features)
                # stoken_features = stoken_features.transpose(-1, -2).reshape(B, C, hh, ww)
            else:
                stoken_features = self.stoken_refine(stoken_features)
            
        # 727
        
        stoken_features = self.unfold(stoken_features) # (B, C*9, hh*ww)
        stoken_features = stoken_features.transpose(1, 2).reshape(B, hh*ww, C, 9) # (B, hh*ww, C, 9)
        # 714
        pixel_features = stoken_features @ affinity_matrix.transpose(-1, -2) # (B, hh*ww, C, h*w)
        # 687
        pixel_features = pixel_features.reshape(B, hh, ww, C, h, w).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        
        # 681
        # 591 for 2 iters
                
        if pad_r > 0 or pad_b > 0:
            pixel_features = pixel_features[:, :, :H0, :W0]
        
        return pixel_features
    
    
    def direct_forward(self, x):
        B, C, H, W = x.shape
        stoken_features = x
        if self.refine:
            if self.refine_attention:
                # stoken_features = stoken_features.flatten(2).transpose(-1, -2)
                stoken_features = self.stoken_refine(stoken_features)
                # stoken_features = stoken_features.transpose(-1, -2).reshape(B, C, H, W)
            else:
                stoken_features = self.stoken_refine(stoken_features)
        return stoken_features
        
    def forward(self, x):
        if self.stoken_size[0] > 1 or self.stoken_size[1] > 1:
            return self.stoken_forward(x)
        else:
            return self.direct_forward(x)


class StokenAttentionLayer(nn.Module):
    def __init__(self, dim, n_iter, stoken_size, 
                 num_heads=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, layerscale=False, init_values = 1.0e-5):
        super().__init__()
                        
        self.layerscale = layerscale
        
        self.pos_embed = ResDWC(dim, 3)
                                        
        self.norm1 = LayerNorm2d(dim)
        self.attn = StokenAttention(dim, stoken_size=stoken_size, 
                                    n_iter=n_iter,                                     
                                    num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                    attn_drop=attn_drop, proj_drop=drop)   
                    
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp2 = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim, act_layer=act_layer, drop=drop)
                
        if layerscale:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(1, dim, 1, 1),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(1, dim, 1, 1),requires_grad=True)
        
    def forward(self, x):
        x = self.pos_embed(x)
        if self.layerscale:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp2(self.norm2(x))) 
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp2(self.norm2(x)))        
        return x


# class SlotAttention(nn.Module):
#     """
#     Slot Attention module from Locatello et al. (NeurIPS 2020).

#     Args:
#         num_slots: int, number of slots (K).
#         dim: int, dimensionality of input features and slot vectors (E).
#         iters: int, number of attention iterations (T).
#         eps: float, numerical stability constant.
#     """
#     def __init__(self, num_slots: int, dim: int, iters: int = 3, eps: float = 1e-8):
#         super(SlotAttention, self).__init__()
#         self.num_slots = num_slots
#         self.dim = dim
#         self.iters = iters
#         self.eps = eps

#         # Parameters for slot initialization (learnable)
#         self.slots_mu = nn.Parameter(torch.randn(1, num_slots, dim))
#         self.slots_sigma = nn.Parameter(torch.abs(torch.randn(1, num_slots, dim)))

#         # Linear maps for computing queries, keys, and values
#         self.to_q = nn.Linear(dim, dim)
#         self.to_k = nn.Linear(dim, dim)
#         self.to_v = nn.Linear(dim, dim)

#         # GRU for slot update
#         self.gru = nn.GRUCell(dim, dim)
#         # MLP for slot refinement
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, dim * 2),
#             nn.ReLU(),
#             nn.Linear(dim * 2, dim)
#         )
#         self.norm_inputs = nn.LayerNorm(dim)
#         self.norm_slots = nn.LayerNorm(dim)
#         self.norm_mlp = nn.LayerNorm(dim)

#     def forward(self, inputs: torch.Tensor):
#         """
#         inputs: torch.Tensor of shape (B, N, E)
#         returns:
#             slots: (B, K, E)
#             attn: (B, N, K)
#         """
#         B, N, E = inputs.shape
#         assert E == self.dim, "Input feature dimension must match slot dimension"

#         # Normalize inputs
#         x = self.norm_inputs(inputs)
#         # Compute keys and values
#         k = self.to_k(x)   # (B, N, E)
#         v = self.to_v(x)   # (B, N, E)

#         # Initialize slots from learned Gaussian parameters
#         slots = self.slots_mu + torch.randn(B, self.num_slots, self.dim, device=inputs.device) * self.slots_sigma
#         slots = slots.to(x.dtype)  # Ensure same dtype as inputs

#         for _ in range(self.iters):
#             slots_prev = slots
#             slots_norm = self.norm_slots(slots)
#             q = self.to_q(slots_norm)  # (B, K, E)
#             # Scaled dot-product attention
#             q = q * (self.dim ** -0.5)

#             # Compute attention logits: (B, N, K)
#             logits = torch.einsum('bne,bke->bnk', k, q)
#             # Attention weights over slots for each input
#             attn = F.softmax(logits, dim=-1) + self.eps  # (B, N, K)
#             # Normalize over inputs for each slot
#             attn_norm = attn / torch.sum(attn, dim=1, keepdim=True)

#             # Weighted mean: (B, K, E)
#             updates = torch.einsum('bnk,bne->bke', attn_norm, v)

#             # Slot update via GRU
#             slots = self.gru(
#                 updates.view(-1, E),
#                 slots_prev.view(-1, E)
#             )
#             slots = slots.view(B, self.num_slots, E)

#             # MLP refinement
#             slots = slots + self.mlp(self.norm_mlp(slots))

#         # Return final slots and last attention (inputs-to-slots)
#         return slots, attn




class SlotAttention(nn.Module):
    def __init__(self, num_slots: int, dim: int, iters: int = 3, eps: float = 1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters
        self.eps = eps

        # Learnable Gaussian initialization parameters
        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, dim))
        self.slots_sigma = nn.Parameter(torch.abs(torch.randn(1, num_slots, dim)))

        # Projection layers
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        # GRU + MLP for slot updates
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )

        # LayerNorms
        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)


    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs: Tensor of shape (B, N, E)
        Returns:
            slots: (B, K, E)
            attn:  (B, N, K)
        """
        B, N, E = inputs.shape
        x = self.norm_inputs(inputs)
        k = self.to_k(x)
        v = self.to_v(x)

        # Sample initial slots from Gaussian
        slots = self.slots_mu + torch.randn(B, self.num_slots, self.dim, device=x.device, dtype=x.dtype) * self.slots_sigma

        for _ in range(self.iters):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)
            q = self.to_q(slots_norm) * (self.dim ** -0.5)

            logits = torch.einsum('bne,bke->bnk', k, q)
            attn = torch.softmax(logits, dim=-1) + self.eps
            attn_norm = attn / torch.sum(attn, dim=1, keepdim=True)

            updates = torch.einsum('bnk,bne->bke', attn_norm, v)

            # GRU step
            slots = self.gru(updates.view(-1, E), slots_prev.view(-1, E))
            slots = slots.view(B, self.num_slots, E)

            # MLP refinement
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots, attn

class SlotSampler(nn.Module):
    def __init__(self, embed_dim: int, num_samples: int, iters: int = 3, eps: float = 1e-8):
        super().__init__()
        self.slot_attn = SlotAttention(
            num_slots=num_samples,
            dim=embed_dim,
            iters=iters,
            eps=eps
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        slots, _ = self.slot_attn(x)
        return slots

        

class MultiStokenWithSampler(nn.Module):
    def __init__(
        self,
        dim: int,
        # stoken-attention layer args:
        n_iter: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        qk_scale = None,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        layerscale: bool = False,
        init_values: float = 1e-5,
        # sampler args:
        sampler_k: int = 4,
        slot_iters: int = 3,
        slot_eps: float = 1e-8
    ):
        super().__init__()
        # three stacked StokenAttentionLayers

        self.stoken_sizes = [(8, 8), (4, 4), (1, 1)]
        self.layer1 = StokenAttentionLayer(
            dim, n_iter, self.stoken_sizes[0],
            num_heads, mlp_ratio, qkv_bias, qk_scale,
            drop, attn_drop, drop_path, act_layer=nn.GELU,
            layerscale=layerscale, init_values=init_values
        )
        self.layer2 = StokenAttentionLayer(
            dim, n_iter, self.stoken_sizes[1],
            num_heads, mlp_ratio, qkv_bias, qk_scale,
            drop, attn_drop, drop_path, act_layer=nn.GELU,
            layerscale=layerscale, init_values=init_values
        )
        self.layer3 = StokenAttentionLayer(
            dim, n_iter, self.stoken_sizes[2],
            num_heads, mlp_ratio, qkv_bias, qk_scale,
            drop, attn_drop, drop_path, act_layer=nn.GELU,
            layerscale=layerscale, init_values=init_values
        )
        # then your SlotSampler:
        self.sampler = SlotSampler(
            embed_dim=dim,
            num_samples=sampler_k,
            iters=slot_iters,
            eps=slot_eps
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: either (B, C, H, W) or (B, N, E). 
        If spatial, we’ll apply the layers in spatial form,
        then flatten to (B, N, E) for the sampler.
        """
        # if sequence flatten, reshape to spatial
        is_seq = (x.dim() == 3)
        if is_seq:
            B, N, E = x.shape
            H = W = int(math.sqrt(N))
            x = x.transpose(1, 2).view(B, E, H, W).contiguous()

        # --- 3 super-token–style attention layers ---
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # flatten back to (B, N, E)
        B, C, H, W = x.shape
        tokens = x.view(B, C, H*W).permute(0, 2, 1)  # (B, N, E)

        # --- finally slot-sample ---
        slots = self.sampler(tokens)  # (B, K, E)
        return slots
    

class StokenAttentionLayerWithSampler(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0.1,
        num_samples: int = 4,
        slot_iters: int = 3,
        slot_eps: float = 1e-8,
    ):
        super().__init__()

        self.slot_sampler = SlotSampler(
            embed_dim=embed_dim,
            num_samples=num_samples,
            iters=slot_iters,
            eps=slot_eps
        )

        #self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, N, E = x.shape

        # Sample slots
        slots = self.slot_sampler(x)  # (B, K, E)

        # Cross-attention: tokens attend to slots
        # attn_out, _ = self.attn(query=x, key=slots, value=slots)  # (B, N, E)

        # Residual + Norm
        # x = x + self.dropout(attn_out)
        x = self.norm1(slots)

        # Feedforward
        ff_out = self.ffn(x)
        x = x + self.dropout(ff_out)

        return x
# Debug examples