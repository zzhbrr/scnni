'''
Author: zzh
Date: 2023-03-09
LastEditTime: 2023-03-09 13:09:21
Description: 
FilePath: /SCNNI/python_scripts/test_net.py
'''
import torch
import torch.nn as nn

class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(x)


if __name__ == '__main__':
    net = TestNet()
    net.eval()
    # x = torch.tensor([1, 2, 3, -1, -2, 4, 5, 6, -7, -8, 12, 11])
    # x = torch.reshape(x, (1, 3, 2, 2))
    x = torch.rand(1, 3, 2, 2)
    with torch.no_grad():
        mod = torch.jit.trace(net, x)
        mod.save("relu_only_net/relu_only_net.pt")