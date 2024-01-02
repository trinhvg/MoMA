
import math
import torch
import torch.nn as nn
from functools import partial, reduce
from operator import mul


from timm.models.vision_transformer import VisionTransformer, _cfg, _create_vision_transformer
from timm.models.layers.helpers import to_2tuple
from timm.models.layers import PatchEmbed

__all__ = [
    'deit_base_patch16_384',
]

def vit_tiny_patch16_224(num_classes=3, pretrained=False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16)
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    model.head = nn.Linear(in_features=768, out_features=num_classes, bias=True)
    nn.init.zeros_(model.head.weight)
    nn.init.constant_(model.head.bias, 0.)
    return model


def vit_tiny_patch16_384(num_classes=3, pretrained=False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16) @ 384x384.
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_384', pretrained=pretrained, **model_kwargs)
    model.head = nn.Linear(in_features=768, out_features=num_classes, bias=True)
    nn.init.zeros_(model.head.weight)
    nn.init.constant_(model.head.bias, 0.)
    return model


def vit_base_patch16_224(num_classes=3, pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    model.head = nn.Linear(in_features=768, out_features=num_classes, bias=True)
    nn.init.zeros_(model.head.weight)
    nn.init.constant_(model.head.bias, 0.)
    return model


def vit_base_patch16_384(num_classes=3, pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_384', pretrained=pretrained, **model_kwargs)
    model.head = nn.Linear(in_features=768, out_features=num_classes, bias=True)
    nn.init.zeros_(model.head.weight)
    nn.init.constant_(model.head.bias, 0.)
    return model



def vit_base_patch32_384(num_classes=3, pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_384', pretrained=pretrained, **model_kwargs)
    model.head = nn.Linear(in_features=768, out_features=num_classes, bias=True)
    nn.init.zeros_(model.head.weight)
    nn.init.constant_(model.head.bias, 0.)
    return model


def deit_tiny_patch16_224(pretrained=False, **kwargs):
    """ DeiT-tiny model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('deit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


def deit_small_patch16_224(num_classes=3, pretrained=False, **kwargs):
    """ DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('deit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    model.head = nn.Linear(in_features=768, out_features=num_classes, bias=True)
    nn.init.zeros_(model.head.weight)
    nn.init.constant_(model.head.bias, 0.)
    return model





def deit_base_patch16_224(num_classes=3, pretrained=False, **kwargs):
    """ DeiT base model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('deit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    model.head = nn.Linear(in_features=768, out_features=num_classes, bias=True)
    nn.init.zeros_(model.head.weight)
    nn.init.constant_(model.head.bias, 0.)
    return model


def deit_base_patch16_384(num_classes=3, pretrained=False, **kwargs):
    """ DeiT base model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('deit_base_patch16_384',pretrained=pretrained, **model_kwargs)
    model.head = nn.Linear(in_features=768, out_features=num_classes, bias=True)
    nn.init.zeros_(model.head.weight)
    nn.init.constant_(model.head.bias, 0.)
    return model



# def _test():
#     import os
#     os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'
#     from torchsummary import summary
#
#     net = vit_base_patch16_384(num_classes=3, pretrained=True)
#     net = net.cuda()
#     # y_class = net(torch.randn(4, 3, 224, 224).cuda())
#     # print(y_class.size())
#     # y_class = net(torch.randn(48, 3, 224, 224).cuda())
#     # print(y_class.size())
#
#     # model = net.cuda()
#     summary(net, (3, 384, 384))
# _test()
