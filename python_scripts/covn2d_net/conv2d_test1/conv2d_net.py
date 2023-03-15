'''
Author: zzh
Date: 2023-03-14 
LastEditTime: 2023-03-14 17:43:59
Description: 
FilePath: /SCNNI/python_scripts/covn2d_net/conv2d_test1/conv2d_net.py
'''
import torch
import torch.nn as nn

class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.conv = nn.Conv2d(in_channels=2,
                              out_channels=4,
                              kernel_size=3,
                              stride=(2, 2),
                              padding=(1, 1),
                              bias=True)
    def forward(self, x):
        return self.conv(x)


if __name__ == '__main__':
    net = TestNet()
    state_dict = torch.load('conv2d_test1.pkl')
    net.load_state_dict(state_dict)
    net.eval()
    x = torch.range(0, 49).reshape(1, 2, 5, 5)
    y = net(x)
    print(y)
    print(y.shape) #[1, 4, 3, 3]
    for name,parameters in net.named_parameters():
        print(name,':',parameters)
    # torch.save(net.state_dict(), "conv2d_test1.pkl")
    # with torch.no_grad():
    #     mod = torch.jit.trace(net, x)
    #     mod.save("conv2d_test1.pt")
