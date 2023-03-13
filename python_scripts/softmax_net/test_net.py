'''
Author: zzh
Date: 2023-03-13 12:42:30
LastEditTime: 2023-03-13 12:48:29
Description: 
FilePath: /SCNNI/python_scripts/softmax_net/test_net.py
'''
import torch
import torch.nn as nn

class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
    def forward(self, x):
        return nn.Softmax(dim=1)(x)


if __name__ == '__main__':
    net = TestNet()
    net.eval()
    # x = torch.rand(1, 5)
    # y = net(x)
    with torch.no_grad():
        mod = torch.jit.trace(net, x)
        mod.save("softmax_net.pt")