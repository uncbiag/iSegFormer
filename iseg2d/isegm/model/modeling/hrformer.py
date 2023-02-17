##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: RainbowSecret
## Microsoft Research
## yuyua@microsoft.com, furao17@mails.ucas.ac.cn
## Copyright (c) 2021
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# from .hrformer_helper.backbone_selector import BackboneSelector
from .hrformer_helper.hrt.module_helper import ModuleHelper
from .hrformer_helper.hrt.modules.spatial_ocr_block import SpatialGather_Module, SpatialOCR_Module

from .hrformer_helper.hrt.logger import Logger as Log
from .hrformer_helper.hrt.hrt_backbone import HRTBackbone, HRTBackbone_v2


class BackboneSelector(object):
    def __init__(self, configer):
        self.configer = configer

    def get_backbone(self, **params):
        backbone = self.configer.get("network", "backbone")

        model = None
        # if (
        #     "resnet" in backbone or "resnext" in backbone or "resnest" in backbone
        # ) and "senet" not in backbone:
        #     model = ResNetBackbone(self.configer)(**params)

        if "hrt" in backbone:
            model = HRTBackbone(self.configer)(**params)
            pass

        # elif "hrnet" in backbone:
        #     model = HRNetBackbone(self.configer)(**params)

        # elif "swin" in backbone:
        #     model = SwinTransformerBackbone(self.configer)(**params)

        else:
            Log.error("Backbone {} is invalid.".format(backbone))
            exit(1)

        return model


class HRT_B_OCR_V3(nn.Module):
    def __init__(self, num_classes, in_ch=3, backbone='hrt_base', bn_type="torchbn", pretrained=None):
        super(HRT_B_OCR_V3, self).__init__()
        self.num_classes = num_classes
        self.bn_type = bn_type
        self.backbone = HRTBackbone_v2(backbone, pretrained, in_ch)()

        in_channels = 1170
        hidden_dim = 512
        group_channel = math.gcd(in_channels, hidden_dim)
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_dim,
                kernel_size=7,
                stride=1,
                padding=3,
                groups=group_channel,
            ),
            ModuleHelper.BNReLU(
                hidden_dim, bn_type=self.bn_type
            ),
        )
        self.ocr_gather_head = SpatialGather_Module(self.num_classes)
        self.ocr_distri_head = SpatialOCR_Module(
            in_channels=hidden_dim,
            key_channels=hidden_dim // 2,
            out_channels=hidden_dim,
            scale=1,
            dropout=0.05,
            bn_type=self.bn_type,
        )
        self.cls_head = nn.Conv2d(
            hidden_dim, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.aux_head = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_dim,
                kernel_size=7,
                stride=1,
                padding=3,
                groups=group_channel,
            ),
            ModuleHelper.BNReLU(
                hidden_dim, bn_type=self.bn_type
            ),
            nn.Conv2d(
                hidden_dim,
                self.num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

    def forward(self, x_):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out_aux = self.aux_head(feats)

        feats = self.conv3x3(feats)

        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)

        out = self.cls_head(feats)

        out_aux = F.interpolate(
            out_aux, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        out = F.interpolate(
            out, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        return out_aux, out


class HRT_S_OCR_V2(nn.Module):
    def __init__(self, num_classes, backbone='hrt_small', bn_type="torchbn", pretrained=None):
        super(HRT_S_OCR_V2, self).__init__()
        self.num_classes = num_classes
        self.bn_type = bn_type
        self.backbone = HRTBackbone_v2(backbone, pretrained)()

        in_channels = 480
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.bn_type),
        )
        self.ocr_gather_head = SpatialGather_Module(self.num_classes)
        self.ocr_distri_head = SpatialOCR_Module(
            in_channels=512,
            key_channels=256,
            out_channels=512,
            scale=1,
            dropout=0.05,
            bn_type=self.bn_type,
        )
        self.cls_head = nn.Conv2d(
            512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.aux_head = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.bn_type),
            nn.Conv2d(
                512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True
            ),
        )

    def forward(self, x_):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out_aux = self.aux_head(feats)

        feats = self.conv3x3(feats)

        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)

        out = self.cls_head(feats)

        out_aux = F.interpolate(
            out_aux, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        out = F.interpolate(
            out, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        return out_aux, out


class HRT_SMALL_OCR_V2(nn.Module):
    def __init__(self, configer):
        super(HRT_SMALL_OCR_V2, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get("data", "num_classes")
        self.backbone = BackboneSelector(configer).get_backbone()

        in_channels = 480
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get("network", "bn_type")),
        )
        self.ocr_gather_head = SpatialGather_Module(self.num_classes)
        self.ocr_distri_head = SpatialOCR_Module(
            in_channels=512,
            key_channels=256,
            out_channels=512,
            scale=1,
            dropout=0.05,
            bn_type=self.configer.get("network", "bn_type"),
        )
        self.cls_head = nn.Conv2d(
            512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.aux_head = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get("network", "bn_type")),
            nn.Conv2d(
                512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True
            ),
        )

    def forward(self, x_):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out_aux = self.aux_head(feats)

        feats = self.conv3x3(feats)

        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)

        out = self.cls_head(feats)

        out_aux = F.interpolate(
            out_aux, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        out = F.interpolate(
            out, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        return out_aux, out


class HRT_BASE_OCR_V2(nn.Module):
    def __init__(self, configer):
        super(HRT_BASE_OCR_V2, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get("data", "num_classes")
        self.backbone = BackboneSelector(configer).get_backbone()

        in_channels = 1170
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get("network", "bn_type")),
        )
        self.ocr_gather_head = SpatialGather_Module(self.num_classes)
        self.ocr_distri_head = SpatialOCR_Module(
            in_channels=512,
            key_channels=256,
            out_channels=512,
            scale=1,
            dropout=0.05,
            bn_type=self.configer.get("network", "bn_type"),
        )
        self.cls_head = nn.Conv2d(
            512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.aux_head = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get("network", "bn_type")),
            nn.Conv2d(
                512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True
            ),
        )

    def forward(self, x_):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out_aux = self.aux_head(feats)

        feats = self.conv3x3(feats)

        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)

        out = self.cls_head(feats)

        out_aux = F.interpolate(
            out_aux, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        out = F.interpolate(
            out, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        return out_aux, out


class HRT_SMALL_OCR_V3(nn.Module):
    def __init__(self, configer):
        super(HRT_SMALL_OCR_V3, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get("data", "num_classes")
        self.backbone = BackboneSelector(configer).get_backbone()

        in_channels = 480
        hidden_dim = 512
        group_channel = math.gcd(in_channels, hidden_dim)
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_dim,
                kernel_size=7,
                stride=1,
                padding=3,
                groups=group_channel,
            ),
            ModuleHelper.BNReLU(
                hidden_dim, bn_type=self.configer.get("network", "bn_type")
            ),
        )
        self.ocr_gather_head = SpatialGather_Module(self.num_classes)
        self.ocr_distri_head = SpatialOCR_Module(
            in_channels=hidden_dim,
            key_channels=hidden_dim // 2,
            out_channels=hidden_dim,
            scale=1,
            dropout=0.05,
            bn_type=self.configer.get("network", "bn_type"),
        )
        self.cls_head = nn.Conv2d(
            hidden_dim, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.aux_head = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_dim,
                kernel_size=7,
                stride=1,
                padding=3,
                groups=group_channel,
            ),
            ModuleHelper.BNReLU(
                hidden_dim, bn_type=self.configer.get("network", "bn_type")
            ),
            nn.Conv2d(
                hidden_dim,
                self.num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

    def forward(self, x_):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out_aux = self.aux_head(feats)

        feats = self.conv3x3(feats)

        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)

        out = self.cls_head(feats)

        out_aux = F.interpolate(
            out_aux, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        out = F.interpolate(
            out, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        return out_aux, out


class HRT_BASE_OCR_V3(nn.Module):
    def __init__(self, configer):
        super(HRT_BASE_OCR_V3, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get("data", "num_classes")
        self.backbone = BackboneSelector(configer).get_backbone()

        in_channels = 1170
        hidden_dim = 512
        group_channel = math.gcd(in_channels, hidden_dim)
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_dim,
                kernel_size=7,
                stride=1,
                padding=3,
                groups=group_channel,
            ),
            ModuleHelper.BNReLU(
                hidden_dim, bn_type=self.configer.get("network", "bn_type")
            ),
        )
        self.ocr_gather_head = SpatialGather_Module(self.num_classes)
        self.ocr_distri_head = SpatialOCR_Module(
            in_channels=hidden_dim,
            key_channels=hidden_dim // 2,
            out_channels=hidden_dim,
            scale=1,
            dropout=0.05,
            bn_type=self.configer.get("network", "bn_type"),
        )
        self.cls_head = nn.Conv2d(
            hidden_dim, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.aux_head = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_dim,
                kernel_size=7,
                stride=1,
                padding=3,
                groups=group_channel,
            ),
            ModuleHelper.BNReLU(
                hidden_dim, bn_type=self.configer.get("network", "bn_type")
            ),
            nn.Conv2d(
                hidden_dim,
                self.num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

    def forward(self, x_):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out_aux = self.aux_head(feats)

        feats = self.conv3x3(feats)

        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)

        out = self.cls_head(feats)

        out_aux = F.interpolate(
            out_aux, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        out = F.interpolate(
            out, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        return out_aux, out