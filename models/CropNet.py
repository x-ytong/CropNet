import torch.nn as nn



def kaiming_init(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, dropout=0.5):
        super().__init__()

        stride = 2 if downsample else 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Dropout2d(dropout)
        )

    def forward(self, x):
        return self.block(x)

class CropNet(nn.Module):
    def __init__(self, data_channel=1, classes=5):
        super().__init__()

        self.layer1 = ConvBlock(data_channel, 128, downsample=True, dropout=0.2)
        self.layer2 = ConvBlock(128, 128, dropout=0.3)
        self.layer3 = ConvBlock(128, 128, dropout=0.3)
        self.layer4 = ConvBlock(128, 128, dropout=0.4)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, classes)

        self.apply(kaiming_init)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


