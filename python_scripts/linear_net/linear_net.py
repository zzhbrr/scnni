'''
Author: zzh
Date: 2023-03-13
LastEditTime: 2023-03-14 12:25:19
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
    state_dict = torch.load('linear_net.pkl')
    net.load_state_dict(state_dict)
    net.eval()
    x = torch.range(0, 4)
    y = net(x)
    print(y)
    for name,parameters in net.named_parameters():
        print(name,':',parameters)
    # torch.save(net.state_dict(), "linear_net.pkl")
    # with torch.no_grad():
    #     mod = torch.jit.trace(net, x)
    #     mod.save("linear_net.pt")
# /ws/downloads/pnnx-20230227-ubuntu/pnnx linear_net.pt inputshape=[1, 5]