import torch
import torch.nn as nn
import torch.nn.functional as F
from engine.logger import get_logger

logger = get_logger()

def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs)


def group_weight(weight_group, module, norm_layer, lr):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, norm_layer) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) \
                or isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.Parameter):
            group_decay.append(m)

    assert len(list(module.parameters())) >= len(group_decay) + len(group_no_decay)
    weight_group.append(dict(params=group_decay, lr=lr))
    weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
    return weight_group


def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try: 
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)


# fvcore flops =======================================
def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    assert not with_complex 
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L    
    return flops
  
def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
    return flops


class CMFS(nn.Module):
    def __init__(self, cfg=None, criterion=None, norm_layer=nn.BatchNorm2d):
        super(CMFS, self).__init__()
        self.channels = [96, 192, 384, 768]
        self.norm_layer = norm_layer
        # import backbone and decoder
        if cfg.backbone == 'vmamba_tiny':
            logger.info('Using backbone: Vmamba-tiny-dual')
            from .encoder import vmamba_tiny as backbone
            self.backbone = backbone(config=cfg, norm_fuse=norm_layer)
        else:
            logger.info('Using backbone: Vmamba-tiny-dual')
            from .encoder import vmamba_tiny as backbone
            self.backbone = backbone(config=cfg, norm_fuse=norm_layer)

        logger.info('Using MLP Decoder')
        from .decoder_segment import SegmentTaskHead
        self.decode_head = SegmentTaskHead(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer, embed_dim=cfg.decoder_embed_dim)
        from .decoder_fusion import FusionTaskHead
        self.aux_head = FusionTaskHead(in_channels=self.channels)

        self.criterion = criterion
        if self.criterion:
            self.init_weights(cfg, pretrained=cfg.pretrained_backbone)

    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            logger.info('Loading pretrained model: {}'.format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)
        logger.info('Initing weights ...')
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')
        if self.aux_head:
            init_weight(self.aux_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')
        logger.info('Initing weights ...')

    def encode_decode(self, rgb, modal_x):
        """Encode images with backbone, decode into a semantic segmentation
        map and fusion image of the same size as input."""
        ori_size = rgb.shape
        ori_inputs = [rgb,modal_x]
        x_vision, x_semantic = self.backbone(rgb, modal_x)
                
        out_semantic = self.decode_head.forward(x_semantic)
        out_semantic = F.interpolate(out_semantic, size=ori_size[2:], mode='bilinear', align_corners=False)
        if self.aux_head:
            Fus_img = self.aux_head.forward(x_vision, ori_inputs)
            out = [out_semantic, Fus_img]
            return out
        return out_semantic

    def forward(self, rgb, modal_x, Mask=None, label=None):
        inputs = [rgb, modal_x]

        if self.aux_head:
            # Image Fusion and Segmentation
            outputs = self.encode_decode(rgb, modal_x)
        else:
            # Segmentation only
            outputs = self.encode_decode(rgb, modal_x)
        if label is not None:            
            if self.aux_head:
                loss = self.criterion(inputs, outputs, Mask, label.long())
            else:
                loss = self.criterion(outputs, Mask, label.long())
            return loss
        return outputs

    def flops(self, shape=(3, 600, 800)):
        from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
        import copy
        
        '''
        code from
        https://github.com/MzeroMiko/VMamba/blob/main/classification/models/vmamba.py#L4
        '''
        
        # shape = self.__input_shape__[1:]
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            # "prim::PythonOp.CrossScan": None,
            # "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanNRow": selective_scan_flop_jit,
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = (torch.randn((1, *shape), device=next(model.parameters()).device), torch.randn((1, *shape), device=next(model.parameters()).device))
        print(len(input))
        for i in input:
            print(i.shape)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=input, supported_ops=supported_ops)

        del model, input
        return sum(Gflops.values()) * 1e9
        return f"params {params} GFLOPs {sum(Gflops.values())}"
    