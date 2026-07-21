import torch
import torch.nn as nn

class RNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        res_x = self.relu(x)
        res_x = self.bn1(res_x)
        res_x = self.conv1(res_x)
        
        res_x = self.relu(res_x)
        res_x = self.bn2(res_x)
        res_x = self.conv2(res_x)

        x += res_x
        return x    


class KCModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.block1 = RNBlock(in_channels=8, out_channels=8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.block2 = RNBlock(in_channels=16, out_channels=16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.block3 = RNBlock(in_channels=32, out_channels=32)

        self.globalpoolidk = nn.AdaptiveAvgPool2d((7, 7)) 
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7,     # 56x56 pooled 3 times (/2) = 7x7
                             64)
        self.fc2 = nn.Linear(64, 4)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.block2(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.block3(x)
        x = self.pool(x)

        x = self.globalpoolidk(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)

        logits = self.softmax(x)
        return logits


if __name__ == "__main__":
    model = KCModel()
    # print(model)

    test = torch.randn((1, 1, 56, 56))
    print(model(test).shape)  # should be (1, 4)