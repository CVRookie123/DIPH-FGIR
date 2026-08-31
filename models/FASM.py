class FASM(nn.Module):
    """
    Frequency-aware spatial attention module.

    This module first refines the input feature map with several lightweight
    residual layers, then generates a sequence of spatial attention maps.
    The attention masks are enhanced in the frequency domain and applied
    progressively to suppress dominant low-frequency responses.
    """

    def __init__(
        self,
        block,
        layer,
        att_size=4,
        num_classes=1000,  # Kept for API compatibility.
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 1024
        self.dilation = 1
        self.att_size = att_size

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None or a 3-element tuple, "
                f"got {replace_stride_with_dilation}"
            )

        self.groups = groups
        self.base_width = width_per_group

        self.layer4 = self._make_layer(block, 512, layer, stride=1)

        # Three 1x1 projection heads are used to generate attention cues
        # progressively from the refined feature map.
        self.feature1 = nn.Sequential(
            conv1x1(self.inplanes, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.feature2 = nn.Sequential(
            conv1x1(self.inplanes, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.feature3 = nn.Sequential(
            conv1x1(self.inplanes, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )

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
        # the residual block starts closer to an identity mapping.
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
        att_expansion = 0.25
        layers = []

        layers.append(
            block(
                self.inplanes,
                int(self.inplanes * att_expansion),
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )

        for _ in range(1, blocks):
            layers.append(
                nn.Sequential(
                    conv1x1(self.inplanes, int(self.inplanes * att_expansion)),
                    nn.BatchNorm2d(int(self.inplanes * att_expansion)),
                )
            )
            self.inplanes = int(self.inplanes * att_expansion)
            layers.append(
                block(
                    self.inplanes,
                    int(self.inplanes * att_expansion),
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _mask(self, feature, x):
        """
        Generate a spatial attention mask from the input feature map.

        The attention is computed in the spatial domain, enhanced in the
        frequency domain by suppressing low-frequency components, and then
        transformed back to the spatial domain.
        """
        # Spatial attention summary.
        spatial_attn = feature.mean(1)

        # Frequency-domain decomposition.
        fft_feature = torch.fft.rfft2(spatial_attn)
        magnitude = torch.abs(fft_feature)
        phase = torch.angle(fft_feature)

        # Enhance high-frequency components by suppressing the central
        # low-frequency region in the spectrum.
        _, h, w = magnitude.shape
        mask = torch.ones(h, w, device=x.device)
        mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 0.3
        enhanced_magnitude = magnitude * mask

        # Transform back to the spatial domain.
        enhanced_attn = torch.fft.irfft2(
            enhanced_magnitude * torch.exp(1j * phase)
        )

        # Normalize the attention map to [0, 1].
        attn = (enhanced_attn - enhanced_attn.min()) / (
            enhanced_attn.max() - enhanced_attn.min()
        )
        return attn.unsqueeze(1)

    def _forward_impl(self, x):
        x = self.layer4(x)

        # Generate the first attention cue.
        fea1 = self.feature1(x)
        attn = 2 - self._mask(fea1, x)

        # Apply the first attention mask to suppress dominant responses.
        x = x * attn.expand(-1, self.inplanes, -1, -1)

        # Generate the second attention cue based on the updated feature map.
        fea2 = self.feature2(x)
        attn = 2 - self._mask(fea2, x)

        # Apply the second attention mask.
        x = x * attn.expand(-1, self.inplanes, -1, -1)

        # Generate the third attention cue.
        fea3 = self.feature3(x)

        # Concatenate the three attention-guided features as the final output.
        x = torch.cat([fea1, fea2, fea3], dim=1)
        return x

    def forward(self, x):
        return self._forward_impl(x)