import math
import torch
import torch.nn as nn

import clip
from .vmamba.vmamba import VSSM, LayerNorm2d
from .modules import ChannelEmbed, ResMoEBlock, SME, MIAModule, FSCModule, SS2D
from engine.logger import get_logger

logger = get_logger()

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Backbone_VSSM(VSSM):
    def __init__(self, config, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer="ln", **kwargs):
        kwargs.update(norm_layer=norm_layer)
        super().__init__(**kwargs)
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_extra = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)
            layer_name_extra = f'outnorm_extra{i}'
            self.add_module(layer_name_extra, layer_extra)

        del self.classifier

        self.load_pretrained(pretrained)

        self.FSC = nn.ModuleList([
            FSCModule(
                hidden_dim=self.dims[i//2],
                drop_path=0.,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=4.0,
                d_state=8 * 2 ** (i//2),)
            for i in range(self.num_layers)
        ])

        self.ChannelEmbeds = nn.ModuleList([
            ChannelEmbed(in_channels=self.dims[i] * 2, out_channels=self.dims[i])
            for i in range(self.num_layers)
        ])

        self.MIA = nn.ModuleList([
            MIAModule(
                cfg=config,
                idx=i,
                region_nums=5 if i < 2 else 1,
                H=math.ceil((config.eval_crop_size[0] if not self.training else config.image_height) // 4 / (2 ** i)),
                W=math.ceil((config.eval_crop_size[1] if not self.training else config.image_width) // 4 / (2 ** i)),
                embed_dims=self.dims[i]
            )
            for i in range(self.num_layers)
        ])

        self.ResMoEBlocks = nn.ModuleList([
            nn.Sequential(
                ResMoEBlock(in_ch=self.dims[i // 2], num_experts=3, topk=1, use_shuffle=True),
                SME(in_ch=self.dims[i // 2]),
            )
            for i in range(self.num_layers * 2)
        ])


    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return

        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            pretrained_weights = _ckpt[key]

            new_weights = {}
            for name, param in pretrained_weights.items():
                if name.startswith("layers"):
                    new_weights[name] = param
                    extra_name = name.replace("layers", "layers_extra", 1)
                    new_weights[extra_name] = param
                elif name.startswith("patch_embed"):
                    new_weights[name] = param
                    extra_name = name.replace("patch_embed", "patch_embed_extra", 1)
                    new_weights[extra_name] = param
                else:
                    new_weights[name] = param
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(new_weights, strict=False)
            print(incompatibleKeys)
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, x_rgb, x_e):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        x_rgb = self.patch_embed(x_rgb)
        x_e = self.patch_embed_extra(x_e)

        outs_semantic = []
        outs_vision = []

        for i, (layer, layer_extra) in enumerate(zip(self.layers, self.layers_extra)):
            b, c, h, w = x_rgb.shape
            o_rgb, x_rgb = layer_forward(layer, x_rgb)  # (B, H, W, C)
            o_e, x_e = layer_forward(layer_extra, x_e)

            o_rgb, o_e = self.MIA[i](o_rgb, o_e)
            x_rgb, x_e = layer.downsample(o_rgb), layer_extra.downsample(o_e)

            if i < 2:
                o_rgb = self.FSC[2 * i](o_rgb.permute(0, 2, 3 ,1).view(b, -1, c), (h, w)) + o_rgb
                o_e = self.FSC[2 * i + 1](o_e.permute(0, 2, 3 ,1).view(b, -1, c), (h, w)) + o_e

            o_1 = o_rgb.permute(0, 2, 3, 1)
            o_2 = o_e.permute(0, 2, 3, 1)

            o_fused = self.ChannelEmbeds[i](torch.cat([o_1, o_2], dim=3).permute(0, 3, 1, 2))

            o_fused = self.ResMoEBlocks[2 * i](o_fused)
            o_fused= self.ResMoEBlocks[2 * i + 1](o_fused)

            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out_rgb = norm_layer(o_rgb)
                norm_layer_extra = getattr(self, f'outnorm_extra{i}')
                out_e = norm_layer_extra(o_e)
                if not self.channel_first:
                    out_rgb = out_rgb.permute(0, 3, 1, 2)
                    out_e = out_e.permute(0, 3, 1, 2)

                outs_vision.append(out_rgb.contiguous())
                outs_vision.append(out_e.contiguous())
                outs_semantic.append(o_fused.contiguous())

        if len(self.out_indices) == 0:
            return x_rgb, x_e

        return outs_vision, outs_semantic


class vmamba_tiny(Backbone_VSSM):
    def __init__(self, config=None, channel_first=True, **kwargs):
        super(vmamba_tiny, self).__init__(
            config=config,
            pretrained='./pretrained/vssm1_tiny_0230s_ckpt_epoch_264.pth',
            depths=[2, 2, 8, 2], dims=96, drop_path_rate=0.2,
            patch_size=4, in_chans=3, num_classes=1000,
            ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="silu",
            ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
            ssm_init="v0", forward_type="v05_noz",
            mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
            patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"),
            downsample_version="v3", patchembed_version="v2",
            use_checkpoint=False, posembed=False, imgsize=224,
        )

