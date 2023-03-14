import torch
import torch.nn as nn

class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()

        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_features = 5, out_features = 3, bias=True)
    def forward(self, x):
        x = self.relu(x)
        x = nn.MaxPool2d(2, 2)(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = nn.Softmax(dim=1)(x)
        return x


if __name__ == '__main__':
    net = TestNet()
    # state_dict = torch.load('relu_maxpool_flatten_linear_softmax_net.pkl')
    # net.load_state_dict(state_dict)
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