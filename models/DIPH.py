import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from CSEM import CSEM
from FASM import FASM

MODEL_URLS = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlock(nn.Module):
    """Standard ResNet basic block."""

    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 is not supported in BasicBlock")

        # Both self.conv1 and self.downsample downsample the input when stride != 1.
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    ResNet bottleneck block.

    This implementation follows the torchvision-style ResNet v1.5 design,
    where the downsampling stride is applied in the 3x3 convolution.
    """

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.0)) * groups

        # Both self.conv2 and self.downsample downsample the input when stride != 1.
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet_Backbone(nn.Module):
    """Truncated ResNet backbone that keeps layers up to layer3."""

    def __init__(
        self,
        block,
        layers,
        num_classes=1000,  # Kept for API compatibility.
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(
            3,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        # Zero-initialize the last BN in each residual branch so that
        # residual blocks start closer to identity mappings.
        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, Bottleneck):
                    nn.init.constant_(module.bn3.weight, 0)
                elif isinstance(module, BasicBlock):
                    nn.init.constant_(module.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        ]
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def DIPH_backbone(pretrained=True, progress=True, **kwargs):
    """Build the truncated ResNet backbone."""
    model = ResNet_Backbone(Bottleneck, [3, 4, 6], **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(
            MODEL_URLS["resnet50"],
            progress=progress,
        )
        for name in list(state_dict.keys()):
            if "fc" in name or "layer4" in name:
                state_dict.pop(name)
        model.load_state_dict(state_dict)

    return model


class ResNet_Refine(nn.Module):
    """Refinement module based on ResNet layer4."""

    def __init__(
        self,
        block,
        layer,
        is_local=True,
        num_classes=1000,  # Kept for API compatibility.
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer
        self.inplanes = 1024
        self.dilation = 1

        self.is_local = is_local
        self.groups = groups
        self.base_width = width_per_group

        self.layer4 = self._make_layer(block, 512, layer, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        # Zero-initialize the last BN in each residual branch so that
        # residual blocks start closer to identity mappings.
        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, Bottleneck):
                    nn.init.constant_(module.bn3.weight, 0)
                elif isinstance(module, BasicBlock):
                    nn.init.constant_(module.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        ]
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
            # Insert CSEM after each additional residual block.
            layers.append(CSEM(num_channels=planes * block.expansion))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.layer4(x)

        # Apply global average pooling to obtain the compact feature vector.
        pooled_x = self.avgpool(x)
        pooled_x = torch.flatten(pooled_x, 1)

        if self.is_local:
            return x, pooled_x
        return pooled_x

    def forward(self, x):
        return self._forward_impl(x)


def DIPH_refine(is_local=True, pretrained=True, progress=True, **kwargs):
    """Build the refinement branch."""
    model = ResNet_Refine(Bottleneck, 3, is_local, **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(
            MODEL_URLS["resnet50"],
            progress=progress,
        )
        for name in list(state_dict.keys()):
            if "layer4" not in name:
                state_dict.pop(name)
        model.load_state_dict(state_dict, strict=False)

    return model


def DIPH_attention(att_size=3, pretrained=False, progress=True, **kwargs):
    """Build the attention branch."""
    model = FASM(Bottleneck, 3, att_size=att_size, **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(
            MODEL_URLS["resnet50"],
            progress=progress,
        )
        for name in list(state_dict.keys()):
            if "fc" in name:
                state_dict.pop(name)
        model.load_state_dict(state_dict)

    return model


class DIPH(nn.Module):
    """Main DIPH model."""

    def __init__(
        self,
        code_length=12,
        num_classes=200,
        att_size=3,
        feat_size=2048,
        device="cpu",
        pretrained=True,
    ):
        super().__init__()

        if att_size % 3 != 0:
            raise ValueError("att_size must be divisible by 3.")

        self.backbone = DIPH_backbone(pretrained=pretrained)
        self.refine_global = DIPH_refine(is_local=False, pretrained=pretrained)
        self.refine_local = DIPH_refine(pretrained=pretrained)
        self.attention = DIPH_attention(att_size=att_size, pretrained=pretrained)

        self.hash_layer_active = nn.Sequential(nn.Tanh())
        self.code_length = code_length

        # Allocate code bits to the global branch.
        # For code_length == 32, one extra bit is assigned so that
        # the total number of bits matches the target length.
        if self.code_length != 32:
            self.W_G = nn.Parameter(torch.Tensor(code_length // 2, feat_size))
        else:
            self.W_G = nn.Parameter(torch.Tensor(code_length // 2 + 1, feat_size))
        nn.init.kaiming_uniform_(self.W_G, a=math.sqrt(5))

        # Allocate code bits to the three local branches.
        self.W_L1 = nn.Parameter(torch.Tensor(code_length // 6, feat_size))
        nn.init.kaiming_uniform_(self.W_L1, a=math.sqrt(5))

        self.W_L2 = nn.Parameter(torch.Tensor(code_length // 6, feat_size))
        nn.init.kaiming_uniform_(self.W_L2, a=math.sqrt(5))

        self.W_L3 = nn.Parameter(torch.Tensor(code_length // 6, feat_size))
        nn.init.kaiming_uniform_(self.W_L3, a=math.sqrt(5))

        self.sigmoid_layer = nn.Sigmoid()

        self.bernoulli = torch.distributions.Bernoulli(0.5)
        self.device = device

    def forward(self, x):
        # Shared backbone feature map.
        out = self.backbone(x)
        batch_size, channels, h, w = out.shape

        # Global feature branch.
        global_f = self.refine_global(out)

        # Attention maps for local branches.
        att_map = self.attention(out)  # Shape: [B, att_size, H, W]
        att_size = att_map.shape[1]

        att_map_rep = att_map.unsqueeze(2).repeat(1, 1, channels, 1, 1)
        out_rep = out.unsqueeze(1).repeat(1, att_size, 1, 1, 1)
        out_local = att_map_rep.mul(out_rep)

        # Split attention-guided local features into three groups.
        out_local1 = out_local[:, : att_size // 3, :, :, :].reshape(
            batch_size * att_size // 3,
            channels,
            h,
            w,
        )
        out_local2 = out_local[:, att_size // 3 : att_size * 2 // 3, :, :, :].reshape(
            batch_size * att_size // 3,
            channels,
            h,
            w,
        )
        out_local3 = out_local[:, att_size * 2 // 3 : att_size, :, :, :].reshape(
            batch_size * att_size // 3,
            channels,
            h,
            w,
        )

        local_f1, avg_local_f1 = self.refine_local(out_local1)
        local_f2, avg_local_f2 = self.refine_local(out_local2)
        local_f3, avg_local_f3 = self.refine_local(out_local3)

        # Estimate fusion weights from average responses.
        gap_global = torch.mean(global_f, dim=1, keepdim=True)
        gap_local1 = torch.mean(avg_local_f1, dim=1, keepdim=True)
        gap_local2 = torch.mean(avg_local_f2, dim=1, keepdim=True)
        gap_local3 = torch.mean(avg_local_f3, dim=1, keepdim=True)

        weights_logits = torch.cat(
            [gap_global, gap_local1, gap_local2, gap_local3],
            dim=1,
        )
        weights = F.softmax(weights_logits, dim=1)
        weights_sig = self.sigmoid_layer(weights)

        # Fuse global and local features.
        fused_global = global_f * weights_sig[:, 0:1]
        fused_local1 = avg_local_f1 * weights_sig[:, 1:2]
        fused_local2 = avg_local_f2 * weights_sig[:, 2:3]
        fused_local3 = avg_local_f3 * weights_sig[:, 3:4]

        # Project fused features to hash sub-codes.
        deep_S_G = self.globalMLP(fused_global)
        deep_S_1 = self.localMLP(fused_local1)
        deep_S_2 = self.localMLP(fused_local2)
        deep_S_3 = self.localMLP(fused_local3)

        deep_S = torch.cat([deep_S_G, deep_S_1, deep_S_2, deep_S_3], dim=1)
        ret = self.hash_layer_active(deep_S)

        if self.training:
            global_class = self.classifier_MLP(global_f)
            local_class1 = self.classifier_MLP(avg_local_f1)
            local_class2 = self.classifier_MLP(avg_local_f2)
            local_class3 = self.classifier_MLP(avg_local_f3)
            return ret, local_f1, global_class, local_class1, local_class2, local_class3

        return ret, local_f1


def diph(code_length, num_classes, att_size, feat_size, device, pretrained=False, **kwargs):
    """Factory function for DIPH."""
    model = DIPH(code_length, num_classes, att_size, feat_size, device, pretrained, **kwargs,)
    return model