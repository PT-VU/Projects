import torch.optim
from torch import nn
import tensorflow as tf


class basic_FC (nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(basic_FC, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_channels,int(0.5*in_channels)),
            # nn.ReLU(),
            # nn.Linear(2*in_channels,4*in_channels),
            # nn.Linear(4*in_channels,2*in_channels),
            nn.Linear(int(0.5*in_channels),int(0.25*in_channels)),
            nn.Linear(int(0.25*in_channels),out_channels)
        )

    def forward(self,x):
        y = self.fc(x)
        # y = x.view(-1, 64 * 70 * 70)
        return y

    def initialize(self):  # 初始化模型参数
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data)


