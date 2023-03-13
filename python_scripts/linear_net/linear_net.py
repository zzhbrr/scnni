'''
Author: zzh
Date: 2023-03-13
LastEditTime: 2023-03-13 12:53:46
Description: 
FilePath: /SCNNI/python_scripts/linear_net/linear_net.py
'''
import torch
import torch.nn as nn

class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.linear = nn.Linear(in_features = 5, out_features = 3, bias=True)
    def forward(self, x):
        return self.linear(x)


if __name__ == '__main__':
    net = TestNet()
    net.eval()
    x = torch.rand(1, 10, 5)
    y = net(x)
    print(y.shape)
    for name,parameters in net.named_parameters():
        print(name,':',parameters)
    # with torch.no_grad():
    #     mod = torch.jit.trace(net, x)
    #     mod.save("linear_net.pt")
# /ws/downloads/pnnx-20230227-ubuntu/pnnx linear_net.pt inputshape=[1, 5]