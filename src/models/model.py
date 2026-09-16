import torch
import torch.nn as nn
import torchvision.models as models


class FloodModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        # ---- فرع الـ SAR (2 channels) ----
        self.sar_backbone = models.resnet50(weights='IMAGENET1K_V2')
        self.sar_backbone.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.sar_backbone.fc = nn.Identity()

        # ---- فرع الـ Optical (12 channels) ----
        self.optical_backbone = models.resnet50(weights='IMAGENET1K_V2')
        self.optical_backbone.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.optical_backbone.fc = nn.Identity()

        # نشيل آخر طبقتين (avgpool + fc) من كل backbone عشان نحتفظ بالـ feature map
        # (مش الـ vector المسطح)، لأن الـ decoder محتاج يعرف "فين" مش بس "إيه"
        self.sar_features = nn.Sequential(*list(self.sar_backbone.children())[:-2])
        self.optical_features = nn.Sequential(*list(self.optical_backbone.children())[:-2])
        # The frozen extractors still respond to train()/eval() for BatchNorm behavior.
        for param in self.sar_features.parameters():
            param.requires_grad = False
        for param in self.optical_features.parameters():
            param.requires_grad = False
        # ناتج كل واحد منهم: (batch, 2048, 8, 8) لصورة 256x256

        # ---- الدمج (Fusion) ----
        # بندمج الـ feature maps على مستوى الـ channels مش بعد التسطيح
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(2048 * 2, 512, kernel_size=1),  # يدمج 4096 channel في 512
            nn.ReLU(inplace=True),
        )

        # ---- Decoder: يرجع الصورة تدريجيًا من 8x8 لـ 256x256 ----
        self.decoder = nn.Sequential(
            self._upsample_block(512, 256),   # 8x8   -> 16x16
            self._upsample_block(256, 128),   # 16x16 -> 32x32
            self._upsample_block(128, 64),    # 32x32 -> 64x64
            self._upsample_block(64, 32),     # 64x64 -> 128x128
            self._upsample_block(32, 16),     # 128x128 -> 256x256
            nn.Dropout2d(p=0.3),
        )

        # الطبقة الأخيرة: تحول الـ 16 channel لـ num_classes (3)
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def _upsample_block(self, in_channels, out_channels):
        num_groups = max(group for group in range(1, 9) if out_channels % group == 0)
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, sar, optical):
        sar_feat = self.sar_features(sar)           # (batch, 2048, 8, 8)
        optical_feat = self.optical_features(optical)  # (batch, 2048, 8, 8)

        combined = torch.cat([sar_feat, optical_feat], dim=1)  # (batch, 4096, 8, 8)
        fused = self.fusion_conv(combined)  # (batch, 512, 8, 8)

        decoded = self.decoder(fused)  # (batch, 16, 256, 256)
        out = self.final_conv(decoded)  # (batch, num_classes, 256, 256)

        return out


if __name__ == "__main__":
    model = FloodModel(num_classes=3)
    fake_sar = torch.randn(1, 2, 256, 256)
    fake_optical = torch.randn(1, 12, 256, 256)
    output = model(fake_sar, fake_optical)
    print("Output shape:", output.shape)  # المفروض: torch.Size([1, 3, 256, 256])

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
