import torch
import torch.nn as nn
import torchvision.models as models


class FloodModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        # ---- فرع الـ SAR (2 channels) ----
        self.sar_backbone = models.resnet50(weights='IMAGENET1K_V2')
        # ResNet الأصلي متصمم لـ 3 channels (RGB)، إحنا عندنا 2 بس (VV, VH)
        self.sar_backbone.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.sar_backbone.fc = nn.Identity()  # نشيل آخر طبقة تصنيف، عايزين الـ features بس

        # ---- فرع الـ Optical (12 channels) ----
        self.optical_backbone = models.resnet50(weights='IMAGENET1K_V2')
        self.optical_backbone.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.optical_backbone.fc = nn.Identity()

        # ---- الدمج (Fusion) ----
        # كل فرع بيطلع 2048 رقم (ده حجم الـ features القياسي لـ ResNet-50)
        # بعد الدمج هيبقى عندنا 2048 + 2048 = 4096
        self.fusion = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
        )

        # ---- Segmentation Head ----
        # ده هيبقى مبسط في البداية: بياخد الـ features ويطلع خريطة لكل بيكسل
        # (النسخة دي مبسطة جداً، مش segmentation architecture كامل زي U-Net،
        #  الهدف دلوقتي إننا نتأكد إن الـ pipeline شغال end-to-end بس)
        self.classifier = nn.Linear(512, num_classes * 256 * 256)
        self.num_classes = num_classes

    def forward(self, sar, optical):
        sar_features = self.sar_backbone(sar)          # (batch, 2048)
        optical_features = self.optical_backbone(optical)  # (batch, 2048)

        combined = torch.cat([sar_features, optical_features], dim=1)  # (batch, 4096)
        fused = self.fusion(combined)  # (batch, 512)

        out = self.classifier(fused)  # (batch, num_classes * 256 * 256)
        out = out.view(-1, self.num_classes, 256, 256)  # (batch, 3, 256, 256)

        return out


# تجربة سريعة للتأكد إن الموديل شغال
if __name__ == "__main__":
    model = FloodModel(num_classes=3)

    # نعمل بيانات وهمية بنفس شكل الداتا الحقيقية بتاعتنا
    fake_sar = torch.randn(2, 2, 256, 256)       # batch=2, channels=2
    fake_optical = torch.randn(2, 12, 256, 256)  # batch=2, channels=12

    output = model(fake_sar, fake_optical)
    print("Output shape:", output.shape)
    # المفروض يطبع: torch.Size([2, 3, 256, 256])