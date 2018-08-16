import torch.nn as nn
import torch.nn.functional as F


class Embedder(nn.Module):

    def forward(self, x):
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        x = self.conv1(x)
        x = self.bn1(x)
        F.relu(x, inplace=True)
        x = F.max_pool2d(x, kernel_size=3)
        x = x.view(x.shape[0], -1)
        x = self.fc_emb(x)
        return x

    def __init__(self, embedding_size=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=5)
        self.fc_emb = nn.Linear(in_features=5*8*8, out_features=embedding_size)
