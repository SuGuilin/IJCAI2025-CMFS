import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append('./utils/')

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from dis_mamba.mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from mamba_ssm.modules.mamba_simple import Mamba, VisionMamba
from utils.utils import LayerNorm
from einops import rearrange
import math
from typing import Optional, Callable
from einops import rearrange, repeat
from functools import partial
from torch import Tensor
from typing import Optional, List
from pytorch_wavelets import DWTForward, DWTInverse  # (or import DWT, IDWT)
from transformers import CLIPProcessor, CLIPModel
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


#################################################################################
## Multi-modal Interaction Alignment Module

def channel_shuffle(x, groups=2):
    batch_size, channels, w, h = x.shape
    group_c = channels // groups
    x = x.view(batch_size, groups, group_c, w, h)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, w, h)
    return x


class VisionMambaBlock(nn.Module):
    def __init__(self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm)), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )    
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states#, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
        d_model,
        d_state=16,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        drop_path=0.,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        device=None,
        dtype=None,
        if_bimamba=False,
        bimamba_type="v2",
        if_divide_out=True,
        init_layer_scale=None,
    ):
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
  
    mixer_cls = partial(VisionMamba, d_state=d_state, bimamba_type=bimamba_type, if_divide_out=if_divide_out, init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs)
    block = VisionMambaBlock(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    return block


class BiMambaBlock(nn.Module):
    def __init__(self, dim, init_ratio=0.5):
        super(BiMambaBlock, self).__init__()
        self.encoder_x1 = create_block(dim)
        self.encoder_x2 = create_block(dim)
        self.norm1 = nn.LayerNorm(dim) 
        self.norm2 = nn.LayerNorm(dim) 
        self.ratio = nn.Parameter(torch.tensor(init_ratio))
    
    def forward(self, x, H, W):
        x1, x2 = x
        x1 = channel_shuffle(x1)
        x2 = channel_shuffle(x2)

        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x1_short = x1
        x2_short = x2
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        B, N, C = x1.shape

        x1 = self.encoder_x1(x1) + x1_short
        x2 = self.encoder_x2(x2) + x2_short

        x1 = x1.transpose(1, 2).view(B, C, H, W)
        x2 = x2.transpose(1, 2).view(B, C, H, W)
        x = [x1, x2]
        return x


class MMFIBlock(nn.Module):
    def __init__(self, cfg, idx, N, dim, num_classes, labels, patch_dim, patch_size, gate_embed=32, squeeze_factor=16, reduction=4):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.labels = labels
        self.config = cfg
        self.idx = idx
        self.N = N
        self.num_classes = num_classes
        self.patch_dim = patch_dim
        self.text_prompts = [f"the photo of {cls}" for cls in self.labels]
        self.pool = nn.AdaptiveAvgPool1d(self.patch_dim)
        self.linearMapping = nn.Linear(self.patch_dim, 512)
        self.conv1 = nn.Conv2d(num_classes, gate_embed, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(gate_embed, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

        self.embed_dims = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embed_dims*2, self.embed_dims *2 // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims * 2 // reduction, self.embed_dims))
            for _ in range(2)
        ])
        self.sigmoid = nn.Sigmoid()

        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, 4 * reduction, kernel_size=3, padding=1, stride=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(4 * reduction, 1, 1),
                nn.Sigmoid()
            )
            for _ in range(2)
        ])

        self.gate = nn.Sequential(
            nn.Linear(self.embed_dims * 2, self.embed_dims * 2 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims * 2 // reduction, self.embed_dims),
            nn.Sigmoid())

    def get_clip_text_features(self, device):
        with torch.no_grad():
            inputs = self.clip_processor(text=self.text_prompts, return_tensors="pt", padding=True).to(device)
            text_features = self.clip_model.get_text_features(**inputs)
        return text_features

    def get_patches(self, imgs, patch_size):
        b, c, h, w = imgs.shape
        patches = [] 
        patch_size_h, patch_size_w = patch_size
        for i in range(0, h, patch_size_h):
            for j in range(0, w, patch_size_w):
                patch = imgs[:, :, i:i+patch_size_h, j:j+patch_size_w]
                patches.append(patch)
        return patches
    
    def forward_clip(self, imgs,):
        self.patch_size = (math.ceil(math.ceil((self.config.eval_crop_size[0] if not self.training else self.config.image_height) // 4 / (2 ** self.idx)) / self.N), 
        math.ceil(math.ceil((self.config.eval_crop_size[1] if not self.training else self.config.image_width) // 4 / (2 ** self.idx)) / self.N))
        b, c, h, w = imgs.shape
        patches = self.get_patches(imgs, self.patch_size)
        text_features = self.get_clip_text_features(imgs.device)
        text_features = text_features.to(imgs.device)

        all_patch_sim = []

        for patch in patches:
            patch_image_features = patch.mean(dim=1).flatten(1) 
            patch_image_features = self.linearMapping(self.pool(patch_image_features))
            sim = [F.cosine_similarity(patch_image_features, text_feature.unsqueeze(0)) for text_feature in text_features]

            sim = torch.stack(sim, dim=1).unsqueeze(-1).unsqueeze(-1)
            patch_sim = self.sigmoid(self.conv2(F.relu(self.conv1(sim)))).squeeze(-1).squeeze(-1)
            all_patch_sim.append(patch_sim)

        all_patch_sim = torch.stack(all_patch_sim, dim=-1)
        importanance_map = all_patch_sim.view(b, 1, h // self.patch_size[0], w // self.patch_size[1]).expand(b, c, h // self.patch_size[0], w // self.patch_size[1])
        return importanance_map

    def forward(self, x1, x2, clip_x):
        B, C, H, W = x1.shape
        ## Feature Partition

        region_h = H // self.N
        region_w = W // self.N

        x1 = x1.view(B, C, self.N, region_h, self.N, region_w).permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C, region_h, region_w)
        x2 = x2.view(B, C, self.N, region_h, self.N, region_w).permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C, region_h, region_w)

        x1_flat = x1.flatten(3).transpose(2, 3)  ## B N*N H*W C
        x2_flat = x2.flatten(3).transpose(2, 3)  ## B N*N H*W C
        gated_weight = self.gate(torch.cat((x1_flat, x2_flat), dim=3))  ## B N*N H*W C
        
        gated_weight = gated_weight.reshape(B, self.N*self.N, region_h, region_w, C).permute(0, 1, 4, 2, 3).contiguous()  ## B N*N C H W
        

        gate_map = self.forward_clip(clip_x).unsqueeze(-1).unsqueeze(-1).permute(0, 2, 3, 1, 4, 5).view(B, self.N*self.N, C, 1, 1).contiguous() ## B N*N
        
        x1_tmp = x1
        x2_tmp = x2

        x1 = x1.view(-1, self.embed_dims, region_h, region_w)  ## B*N*N C R_h R_w
        x2 = x2.view(-1, self.embed_dims, region_h, region_w)  ## B*N*N C R_h R_w

        BN, _, _, _ = x1.shape

        ## Hybrid-Att

        gap_x1 = self.avg_pool(x1).view(BN, self.embed_dims)   ## B*N*N C
        gmp_x1 = self.max_pool(x1).view(BN, self.embed_dims)   ## B*N*N C
        ap_x1 = torch.mean(x1, dim=1, keepdim=True)
        mp_x1, _ = torch.max(x1, dim=1, keepdim=True)
        gp_x1 = torch.cat([gap_x1, gmp_x1], dim=1)
        p_x1 = torch.cat([ap_x1, mp_x1], dim=1)
        gp_x1_ca = self.fc[0](gp_x1).view(BN, self.embed_dims, 1, 1)  ## B*N*N C 1 1
        gp_x1_ca = self.sigmoid(gp_x1_ca)
        p_x1_sp = self.mlp[0](p_x1)   ## B*N*N 1 R_h R_w

        gap_x2 = self.avg_pool(x2).view(BN, self.embed_dims)
        gmp_x2 = self.max_pool(x2).view(BN, self.embed_dims)
        ap_x2 = torch.mean(x2, dim=1, keepdim=True)
        mp_x2, _ = torch.max(x2, dim=1, keepdim=True)
        gp_x2 = torch.cat([gap_x2, gmp_x2], dim=1)
        p_x2 = torch.cat([ap_x2, mp_x2], dim=1)
        gp_x2_ca = self.fc[1](gp_x2).view(BN, self.embed_dims, 1, 1)
        gp_x2_ca = self.sigmoid(gp_x2_ca)
        p_x2_sp = self.mlp[1](p_x2)

        out_x1 = (x2 * gp_x1_ca + x2 * p_x1_sp).view(B, -1, C, region_h, region_w) * gated_weight * gate_map + x1_tmp
        out_x2 = (x1 * gp_x2_ca + x1 * p_x2_sp).view(B, -1, C, region_h, region_w) * (1 - gated_weight) * gate_map + x2_tmp

        out_x1 = out_x1.view(B, self.N, self.N, C, region_h, region_w).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        out_x2 = out_x2.view(B, self.N, self.N, C, region_h, region_w).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

        return out_x1, out_x2


class MIAModule(nn.Module):
    def __init__(self, cfg, H, W, idx, region_nums, embed_dims, squeeze_factor=16):
        super().__init__()
        self.N = region_nums
        self.embed_dims = embed_dims
        self.bimamba = BiMambaBlock(dim=embed_dims)
        self.clip_conv = nn.Sequential(
            nn.Conv2d(embed_dims*2, embed_dims*2 // squeeze_factor, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dims*2 // squeeze_factor, embed_dims, 1, bias=False),
        )
        self.MMFI = MMFIBlock(cfg=cfg, idx=idx, N=self.N, dim=embed_dims, num_classes=cfg.num_classes-1, 
                                  labels=cfg.labels, patch_dim=math.ceil(cfg.image_height // 4 / (2 ** idx) / self.N) * math.ceil(cfg.image_width // 4 / (2 ** idx) / self.N), 
                                  patch_size=(math.ceil(H / self.N), math.ceil(W / self.N)), gate_embed=32)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        clip_x = self.clip_conv(torch.cat([x1, x2], dim=1))
        x1, x2 = self.bimamba([x1, x2], H, W)
        out_x1, out_x2 = self.MMFI(x1, x2, clip_x)

        return out_x1, out_x2


#################################################################################
## Frequency-Spatial Collaboration Module

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=12):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        # scan from four direction
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (1, 4, 192, 3136)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class SS2D_local(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def local_scan(self, x, H=14, W=14, w=7, flip=False, column_first=False):
        """Local windowed scan in LocalMamba
        Input:
            x: [B, C, H, W]
            H, W: original width and height
            column_first: column-wise scan first (the additional direction in VMamba)
        Return: [B, C, L]
        """
        B, C, _, _ = x.shape
        x = x.view(B, C, H, W)
        Hg, Wg = math.floor(H / w), math.floor(W / w)

        if H % w != 0 or W % w != 0:
            newH, newW = Hg * w, Wg * w
            x = x[:, :, :newH, :newW]
        if column_first:
            x = x.view(B, C, Hg, w, Wg, w).permute(0, 1, 5, 3, 4, 2).reshape(B, C, -1)
        else:
            x = x.view(B, C, Hg, w, Wg, w).permute(0, 1, 3, 5, 2, 4).reshape(B, C, -1)
        if flip:
            x = x.flip([-1])
        return x

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x1 = self.local_scan(x, H, W, w=4)
        x2 = self.local_scan(x, H, W, w=4, column_first=True)
        x3 = self.local_scan(x, H, W, w=4, flip=True)
        x4 = self.local_scan(x, H, W, w=4, column_first=True, flip=True)

        xs = torch.stack([x1, x2, x3, x4], dim=1)
        # x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        # xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class FSCModule(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.vanilla_attention = SS2D(d_model=hidden_dim, d_state=d_state, expand=expand, dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))

        self.conv_init = nn.Sequential(  # PW->DW->
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.GELU()
        )
        
        self.ln_11 = norm_layer(hidden_dim)
        self.local_attention = SS2D_local(d_model=hidden_dim, d_state=d_state, expand=expand, dropout=attn_drop_rate, **kwargs)
        self.drop_path1 = DropPath(drop_path)
        self.skip_scale1 = nn.Parameter(torch.ones(hidden_dim))

        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

        self.denoise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim*2, hidden_dim*2, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_dim*2, hidden_dim*2, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.dconv = nn.Sequential(
            nn.Conv2d(2*hidden_dim, hidden_dim, 1),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, padding_mode='reflect'),
            nn.GELU()
        )

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim// 4, hidden_dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, input, x_size):
        # x [B,HW,C]
        B, L, C = input.shape
        H, W = x_size
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]

        prepare = rearrange(input, "b h w c -> b c h w").contiguous().to(input.device)
        prepare = self.conv_init(prepare)

        reminder_h = 0
        reminder_w = 0
        if H % 4 != 0 or W % 4 != 0:
            reminder_h = H % 4
            reminder_w = W % 4
            pad = nn.ReplicationPad2d((0, reminder_w, 0, 4 - reminder_h))
            prepare = pad(prepare)

        xfm = DWTForward(J=2, mode='zero', wave='haar').to(input.device)
        ifm = DWTInverse(mode='zero', wave='haar').to(input.device)

        # Yl: (b, c, h/4, w/4)   Yh: 4*(b, 3, h/2, w/2),  4*(b, 3, h/4, w/4)
        Yl, Yh = xfm(prepare)

        h00 = torch.zeros(prepare.shape).float().to(input.device)
        for i in range(len(Yh)):  # len(Yh) = 2
            if i == len(Yh) - 1:  # handle second floor DWT (b, c, h/4, w/4)
                h00[:, :, :Yl.size(2), :Yl.size(3)] = Yl
                h00[:, :, :Yl.size(2), Yl.size(3):Yl.size(3) * 2] = Yh[i][:, :, 0, :, :]
                h00[:, :, Yl.size(2):Yl.size(2) * 2, :Yl.size(3)] = Yh[i][:, :, 1, :, :]
                h00[:, :, Yl.size(2):Yl.size(2) * 2, Yl.size(3):Yl.size(3) * 2] = Yh[i][:, :, 2, :, :]
            else:  # handle first floor DWT (b, c, h/2, w/2)
                h00[:, :, :Yh[i].size(3), Yh[i].size(4):] = Yh[i][:, :, 0, :, :h00.shape[3] - Yh[i].size(4)]
                h00[:, :, Yh[i].size(3):, :Yh[i].size(4)] = Yh[i][:, :, 1, :h00.shape[2] - Yh[i].size(3), :]
                h00[:, :, Yh[i].size(3):, Yh[i].size(4):] = Yh[i][:, :, 2, :h00.shape[2] - Yh[i].size(3),
                                                            :h00.shape[3] - Yh[i].size(4)]

        h00 = rearrange(h00, "b c h w -> b h w c").contiguous()

        h11 = self.ln_11(h00)
        h11 = h00 * self.skip_scale1 + self.drop_path1(self.local_attention(h11))
        h11 = rearrange(h11, "b h w c -> b c h w").contiguous()

        for i in range(len(Yh)):
            if i == len(Yh) - 1:
                Yl = h11[:, :, :Yl.size(2), :Yl.size(3)]
                Yh[i][:, :, 0, :, :] = h11[:, :, :Yl.size(2), Yl.size(3):Yl.size(3) * 2]
                Yh[i][:, :, 1, :, :] = h11[:, :, Yl.size(2):Yl.size(2) * 2, :Yl.size(3)]
                Yh[i][:, :, 2, :, :] = h11[:, :, Yl.size(2):Yl.size(2) * 2, Yl.size(3):Yl.size(3) * 2]
            else:
                Yh[i][:, :, 0, :, :h11.shape[3] - Yh[i].size(4)] = h11[:, :, :Yh[i].size(3), Yh[i].size(4):]
                Yh[i][:, :, 1, :h11.shape[2] - Yh[i].size(3), :] = h11[:, :, Yh[i].size(3):, :Yh[i].size(4)]
                Yh[i][:, :, 2, :h11.shape[2] - Yh[i].size(3), :h11.shape[3] - Yh[i].size(4)] = h11[:, :, Yh[i].size(3):,
                                                                                               Yh[i].size(4):]
                
        Yl = Yl.to(input.device)
        temp = ifm((Yl, [Yh[1]]))
        recons2 = ifm((temp, [Yh[0]])).to(input.device)

        _, _, newh, neww = recons2.shape
        if H % 4 != 0 or W % 4 != 0:
            recons2 = recons2[:, :, 0:newh+reminder_h-4, 0:neww-reminder_w]

        recons2 = rearrange(recons2, "b c h w -> b h w c").contiguous()

        x = self.ln_1(input)
        x = input * self.skip_scale + self.drop_path(self.vanilla_attention(x))
        x = x * self.skip_scale2 + self.drop_path(self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous())

        prepare_fft = torch.cat([x, recons2], dim=3)
        prepare_fft = rearrange(prepare_fft, "b h w c -> b c h w").contiguous().to(input.device)

        input_freq = torch.fft.rfft2(prepare_fft) + 1e-8
        mag = torch.abs(input_freq)
        pha = torch.angle(input_freq)
        mag = self.denoise_conv(mag)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag) + 1e-8
        x_out = torch.fft.irfft2(x_out, s=tuple(x_size), norm='backward') + 1e-8
        x_out = torch.abs(x_out) + 1e-8

        x_out = self.dconv(x_out)
        x_out = self.sca(x_out) * x_out
        x_out = rearrange(x_out, "b c h w -> b h w c").contiguous()

        x = x.view(B, -1, C).contiguous()
        x_out = x_out.view(B, -1, C).contiguous()

        x_final = x_out.view(B, *x_size, C).permute(0, 3, 1, 2).contiguous()
        return x_final


#################################################################################
## Multi-Modal Feature Enhancement Module

class MoEBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        num_experts: int,
        topk: int,
        use_shuffle: bool = True,
        lr_space: str = "linear",
        recursive: int = 2,
    ):
        super().__init__()
        self.use_shuffle = use_shuffle
        self.recursive = recursive

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_ch, 2 * in_ch, kernel_size=1, padding=0),
        )

        self.agg_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=4, stride=4, padding=2, groups=in_ch), nn.GELU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch),
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
        )

        self.conv_2 = nn.Sequential(
            # StripedConv2d(in_ch, kernel_size=3, depthwise=True), 
            nn.LayerNorm(in_ch),
            Mamba(in_ch, d_state=64, bimamba_type=None),
            nn.GELU()
        )

        if lr_space == "linear":
            grow_func = lambda i: i + 2
        elif lr_space == "exp":
            grow_func = lambda i: 2 ** (i + 1)
        elif lr_space == "double":
            grow_func = lambda i: 2 * i + 2
        else:
            raise NotImplementedError(f"lr_space {lr_space} not implemented")

        self.moe_layer = MoELayer(
            experts=[
                Expert(in_ch=in_ch, low_dim=grow_func(i)) for i in range(num_experts)
            ],  # add here multiple of 2 as low_dim
            gate=Router(in_ch=in_ch, num_experts=num_experts),
            num_expert=topk,
        )

        self.proj = nn.Conv2d(in_ch, in_ch, kernel_size=1)

    def calibrate(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        res = x

        for _ in range(self.recursive):
            x = self.agg_conv(x)
        x = self.conv(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return res + x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = self.conv_1(x)

        if self.use_shuffle:
            x = channel_shuffle(x, groups=2)
        x, k = torch.chunk(x, chunks=2, dim=1)

        x = self.conv_2(x.permute(0, 2, 3, 1).view(b, -1, c)).view(b, h, w, c).permute(0, 3, 1, 2)
        k = self.calibrate(k)

        x = self.moe_layer(x, k)
        x = self.proj(x)
        return x


class MoELayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, num_expert: int = 1):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_expert = num_expert

    def forward(self, inputs: torch.Tensor, k: torch.Tensor):
        out = self.gate(inputs)
        weights = F.softmax(out, dim=1, dtype=torch.float).to(inputs.dtype)
        topk_weights, topk_experts = torch.topk(weights, self.num_expert)
        # normalize the weights of the selected experts
        # topk_weights = F.softmax(topk_weights, dim=1, dtype=torch.float).to(inputs.dtype)
        out = inputs.clone()

        if self.training:
            exp_weights = torch.zeros_like(weights)
            exp_weights.scatter_(1, topk_experts, weights.gather(1, topk_experts))
            for i, expert in enumerate(self.experts):
                out += expert(inputs, k) * exp_weights[:, i : i + 1, None, None]
        else:
            selected_experts = [self.experts[i] for i in topk_experts.squeeze(dim=0)]
            for i, expert in enumerate(selected_experts):
                out += expert(inputs, k) * topk_weights[:, i : i + 1, None, None]

        return out

# Reference: https://github.com/eduardzamfir/seemoredetails/blob/main/basicsr/archs/seemore_arch.py
class Expert(nn.Module):
    def __init__(
        self,
        in_ch: int,
        low_dim: int,
    ):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_ch, low_dim, kernel_size=1, padding=0)
        self.conv_2 = nn.Conv2d(in_ch, low_dim, kernel_size=1, padding=0)
        self.conv_3 = nn.Conv2d(low_dim, in_ch, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x = self.conv_2(k) * x  # here no more sigmoid
        x = self.conv_3(x)
        return x


class Router(nn.Module):
    def __init__(self, in_ch: int, num_experts: int):
        super().__init__()

        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange("b c 1 1 -> b c"),
            nn.Linear(in_ch, num_experts, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class StripedConv2d(nn.Module):
    def __init__(self, in_ch: int, kernel_size: int, depthwise: bool = False):
        super().__init__()
        self.in_ch = in_ch
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_ch,
                in_ch,
                kernel_size=(1, self.kernel_size),
                padding=(0, self.padding),
                groups=in_ch if depthwise else 1,
            ),
            nn.Conv2d(
                in_ch,
                in_ch,
                kernel_size=(self.kernel_size, 1),
                padding=(self.padding, 0),
                groups=in_ch if depthwise else 1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class GatedFFN(nn.Module):
    def __init__(self, in_ch, mlp_ratio, kernel_size, act_layer,):
        super().__init__()
        mlp_ch = in_ch * mlp_ratio
        
        self.fn_1 = nn.Sequential(
            nn.Conv2d(in_ch, mlp_ch, kernel_size=1, padding=0),
            act_layer,
        )
        self.fn_2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0),
            act_layer,
        )
        
        self.gate = nn.Conv2d(mlp_ch // 2, mlp_ch // 2, 
                              kernel_size=kernel_size, padding=kernel_size // 2, groups=mlp_ch // 2)

    def feat_decompose(self, x):
        s = x - self.gate(x)
        x = x + self.sigma * s
        return x
    
    def forward(self, x: torch.Tensor):
        x = self.fn_1(x)
        x, gate = torch.chunk(x, 2, dim=1)
        
        gate = self.gate(gate)
        x = x * gate
        
        x = self.fn_2(x)
        return x


class StripedConvFormer(nn.Module):
    def __init__(self, in_ch: int, kernel_size: int):
        super().__init__()
        self.in_ch = in_ch
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        self.proj = nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0)
        self.to_qv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 2, kernel_size=1, padding=0),
            nn.GELU(),
        )

        self.attn = StripedConv2d(in_ch, kernel_size=kernel_size, depthwise=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, v = self.to_qv(x).chunk(2, dim=1)
        q = self.attn(q)
        x = self.proj(q * v)
        return x


class SME(nn.Module):
    def __init__(self, in_ch: int, kernel_size: int = 11):
        super().__init__()
        
        self.norm_1 = LayerNorm(in_ch, data_format='channels_first')
        self.block = StripedConvFormer(in_ch=in_ch, kernel_size=kernel_size)
    
        self.norm_2 = LayerNorm(in_ch, data_format='channels_first')
        self.ffn = GatedFFN(in_ch, mlp_ratio=2, kernel_size=3, act_layer=nn.GELU())
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(self.norm_1(x)) + x
        x = self.ffn(self.norm_2(x)) + x
        return x


class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(ChannelEmbed, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels//reduction, kernel_size=1, bias=True),
                        nn.Conv2d(out_channels//reduction, out_channels//reduction, kernel_size=3, stride=1, padding=1, bias=True, groups=out_channels//reduction),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channels//reduction, out_channels, kernel_size=1, bias=True),
                        norm_layer(out_channels) 
                        )
        self.norm = norm_layer(out_channels)
        
    def forward(self, x):
        residual = self.residual(x)
        x = self.channel_embed(x)
        out = self.norm(residual + x)
        return out


class ResMoEBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        num_experts: int,
        topk: int,
        lr_space: int = 2,
        recursive: int = 2,
        use_shuffle: bool = False,
    ):
        super().__init__()
        lr_space_mapping = {1: "linear", 2: "exp", 3: "double"}
        self.norm = LayerNorm(in_ch, data_format="channels_first")
        self.block = MoEBlock(
            in_ch=in_ch,
            num_experts=num_experts,
            topk=topk,
            use_shuffle=use_shuffle,
            recursive=recursive,
            lr_space=lr_space_mapping.get(lr_space, "linear"),
        )
        self.norm_2 = LayerNorm(in_ch, data_format='channels_first')
        self.ffn = GatedFFN(in_ch, mlp_ratio=2, kernel_size=3, act_layer=nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(self.norm(x)) + x
        x = self.ffn(self.norm_2(x)) + x
        return x


## Feature Modulation
class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=True):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.MLP = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.LeakyReLU(),
            nn.Linear(in_channels * 2, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, text_embed):
        text_embed = text_embed.unsqueeze(1)
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.MLP(text_embed).view(batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        return x
