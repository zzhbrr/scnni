'''
Author: zzh
Date: 2023-03-14
LastEditTime: 2023-03-15 03:09:15
Description: 
FilePath: /scnni/python_scripts/relu_maxpool_flatten_linear_softmax_net/relu_maxpool_flatten_linear_softmax_net.py
'''
import torch
import torch.nn as nn

class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()

        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_features = 12, out_features = 3, bias=True)
    def forward(self, x):
        y = self.relu(x)
        y = nn.MaxPool2d(2, 2)(y)
        y = torch.flatten(y, 1)
        y = self.linear(y)
        y = nn.Softmax(dim=1)(y)
        return y


if __name__ == '__main__':
    net = TestNet()
    state_dict = torch.load('relu_maxpool_flatten_linear_softmax_net.pkl')
    net.load_state_dict(state_dict)
    net.eval()

    # x = torch.tensor([1, 2, 3, -1, -2, 4, 5, 6, -7, -8, 12, 11])
    # x = torch.reshape(x, (1, 3, 2, 2))
    x = torch.rand(1, 3, 4, 4)
    y = net(x)
    # print(y.shape)
    torch.save(net.state_dict(), "relu_maxpool_flatten_linear_softmax_net.pkl")
    with torch.no_grad():
        mod = torch.jit.trace(net, x)
        mod.save("relu_maxpool_flatten_linear_softmax_net.pt")
