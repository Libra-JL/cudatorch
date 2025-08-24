from torch import Tensor, nn


class NiNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, strides, padding):
        super().__init__()
        self.block = nn.Sequential( #  [1,1,224,224]  (224-11)/2+1
            nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x) -> Tensor:
        return self.block(x)




class NiN(nn.Module):
    def __init__(self, num_classes: int):
        super(NiN, self).__init__()
        self.extractor = nn.Sequential(
            NiNBlock(in_channels=1, out_channels=96, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2d(3, stride=2),

            NiNBlock(in_channels=96, out_channels=256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(3, stride=2),

            NiNBlock(in_channels=256, out_channels=384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2d(3, stride=2),

            nn.Dropout(0.5)
        )
        self.classifier = nn.Sequential(
            NiNBlock(in_channels=384, out_channels=num_classes, kernel_size=3, strides=1, padding=1),
            nn.AdaptiveAvgPool2d(output_size=1),  # 等价为：nn.AvgPool2d(kernel_size=5, stride=5, padding=0)
            nn.Flatten()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """
        默认使用的 Kaiming 均匀分布初始化，似乎难以训练模型
        使用 Kaiming 正态分布初始化
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x: Tensor) -> Tensor:
        x = self.extractor(x)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    from torchinfo import summary

    model = NiN(num_classes=10)
    summary(model, input_size=(1, 1, 224, 224))