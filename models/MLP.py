import torch.nn as nn



class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()

        dims = [in_dim, 128, 128, 128, 128, 128, 128, 128, 128, num_classes]

        layers = []
        for i in range(len(dims) - 2):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        layers += [nn.Linear(dims[-2], dims[-1])]

        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (N, in_dim)
        return self.net(x)

