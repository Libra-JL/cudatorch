from torch import nn, Tensor
import torch
from VGGBlock import VGGBlock

from torchinfo import summary


class VGG11(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.extractor = nn.Sequential(
            # 输出通道数由 16 逐层翻倍至 128
            VGGBlock(1, 16),  # 灰度图
            VGGBlock(16, 32),
            VGGBlock(32, 64),
            VGGBlock(64, 128),
            VGGBlock(128, 128),
        )

        self.classifier = nn.Sequential(
            # 输入图像为 224x224，经过 5 个 VGGBlock 后尺寸变为 7x7
            nn.Linear(7 * 7 * 128, 4096), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x) -> Tensor:
        x = self.extractor(x)
        x = torch.flatten(x, 1)  # 展平，准备进入全连接层
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = VGG11(10)
    summary(model, (1, 1, 224, 224))  #这个真好使
