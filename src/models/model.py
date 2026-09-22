import torch
import torch.nn as nn
import torchvision.models as models


class FloodModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        # ---- فرع الـ SAR (2 channels) ----
        sar_backbone = models.resnet50(weights='IMAGENET1K_V2')
        sar_backbone.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.sar_stem, self.sar_layer1, self.sar_layer2, self.sar_layer3, self.sar_layer4 = (
            self._split_resnet_stages(sar_backbone)
        )

        # ---- فرع الـ Optical (3-channel S2RGB, native ImageNet conv1) ----
        optical_backbone = models.resnet50(weights='IMAGENET1K_V2')
        # Keep ResNet-50's native pretrained 3-channel conv1 for S2RGB input.
        self.optical_stem, self.optical_layer1, self.optical_layer2, self.optical_layer3, self.optical_layer4 = (
            self._split_resnet_stages(optical_backbone)
        )

        for module in (
            self.sar_stem, self.sar_layer1, self.sar_layer2, self.sar_layer3, self.sar_layer4,
            self.optical_stem, self.optical_layer1, self.optical_layer2, self.optical_layer3, self.optical_layer4,
        ):
            for param in module.parameters():
                param.requires_grad = False
        # ناتج المراحل (لصورة 256x256): layer1 64x64/256ch، layer2 32x32/512ch،
        # layer3 16x16/1024ch، layer4 8x8/2048ch

        # ---- الدمج (Fusion) عند أعمق مستوى 8x8 ----
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(2048 * 2, 512, kernel_size=1),  # يدمج 4096 channel في 512
            nn.ReLU(inplace=True),
        )
        # Skip projections: concatenated SAR+Optical at 64/32/16, then 1x1 reduce
        self.skip_proj_64 = nn.Conv2d(256 * 2, 64, kernel_size=1)    # 64x64
        self.skip_proj_32 = nn.Conv2d(512 * 2, 128, kernel_size=1)   # 32x32
        self.skip_proj_16 = nn.Conv2d(1024 * 2, 256, kernel_size=1)  # 16x16

        # ---- Decoder: U-Net skips at 16/32/64, then plain upsample to 256 ----
        self.up_8_to_16 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.dec_norm_16 = self._norm_relu(256 + 256)  # after cat skip_16
        self.up_16_to_32 = nn.ConvTranspose2d(256 + 256, 128, kernel_size=4, stride=2, padding=1)
        self.dec_norm_32 = self._norm_relu(128 + 128)  # after cat skip_32
        self.up_32_to_64 = nn.ConvTranspose2d(128 + 128, 64, kernel_size=4, stride=2, padding=1)
        self.dec_norm_64 = self._norm_relu(64 + 64)    # after cat skip_64
        self.up_64_to_128 = self._upsample_block(64 + 64, 32)   # 64x64 -> 128x128
        self.up_128_to_256 = self._upsample_block(32, 16)        # 128x128 -> 256x256
        self.dropout = nn.Dropout2d(p=0.3)

        # الطبقة الأخيرة: تحول الـ 16 channel لـ num_classes (3)
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    @staticmethod
    def _split_resnet_stages(backbone):
        stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        return stem, backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4

    def _norm_relu(self, channels):
        num_groups = max(group for group in range(1, 9) if channels % group == 0)
        return nn.Sequential(
            nn.GroupNorm(num_groups, channels),
            nn.ReLU(inplace=True),
        )

    def _upsample_block(self, in_channels, out_channels):
        num_groups = max(group for group in range(1, 9) if out_channels % group == 0)
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def _encode(self, x, stem, layer1, layer2, layer3, layer4):
        x = stem(x)
        f1 = layer1(x)
        f2 = layer2(f1)
        f3 = layer3(f2)
        f4 = layer4(f3)
        return f1, f2, f3, f4

    def forward(self, sar, optical):
        sar_f1, sar_f2, sar_f3, sar_f4 = self._encode(
            sar, self.sar_stem, self.sar_layer1, self.sar_layer2, self.sar_layer3, self.sar_layer4
        )
        opt_f1, opt_f2, opt_f3, opt_f4 = self._encode(
            optical, self.optical_stem, self.optical_layer1, self.optical_layer2, self.optical_layer3, self.optical_layer4
        )

        fused = self.fusion_conv(torch.cat([sar_f4, opt_f4], dim=1))  # (batch, 512, 8, 8)
        skip_16 = self.skip_proj_16(torch.cat([sar_f3, opt_f3], dim=1))  # (batch, 256, 16, 16)
        skip_32 = self.skip_proj_32(torch.cat([sar_f2, opt_f2], dim=1))  # (batch, 128, 32, 32)
        skip_64 = self.skip_proj_64(torch.cat([sar_f1, opt_f1], dim=1))  # (batch, 64, 64, 64)

        x = self.up_8_to_16(fused)
        x = self.dec_norm_16(torch.cat([x, skip_16], dim=1))
        x = self.up_16_to_32(x)
        x = self.dec_norm_32(torch.cat([x, skip_32], dim=1))
        x = self.up_32_to_64(x)
        x = self.dec_norm_64(torch.cat([x, skip_64], dim=1))
        x = self.up_64_to_128(x)
        x = self.up_128_to_256(x)
        x = self.dropout(x)
        out = self.final_conv(x)

        return out


if __name__ == "__main__":
    model = FloodModel(num_classes=3)
    fake_sar = torch.randn(1, 2, 256, 256)
    fake_optical = torch.randn(1, 3, 256, 256)
    output = model(fake_sar, fake_optical)
    print("Output shape:", output.shape)  # المفروض: torch.Size([1, 3, 256, 256])

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
