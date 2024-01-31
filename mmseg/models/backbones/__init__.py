# Copyright (c) OpenMMLab. All rights reserved.
from .beit import BEiT
from .resnet import ResNet, ResNetV1c, ResNetV1d
from .beit_adapter import BEiT_adapter


__all__ = [
    'ResNet', 'BEiT','BEiT_adapter'
]
