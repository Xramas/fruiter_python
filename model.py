import torch.nn as nn

class Net(nn.Module):
    def __init__(self, num_classes=3):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            # ... (前面的卷积层保持不变)
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 移除最后的 MaxPool2d，用 AdaptiveAvgPool2d 替换
            # 这会输出一个 128x4x4 的张量，如果输入是64x64
            # 为了更通用，我们使用 (1, 1) 的输出尺寸
            nn.AdaptiveAvgPool2d((1, 1)) 
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            # 输入维度现在固定为128，因为 AdaptiveAvgPool2d((1, 1)) 的输出是 128x1x1
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        # 展平操作现在更简单
        x = x.view(x.size(0), -1) 
        return self.classifier(x)