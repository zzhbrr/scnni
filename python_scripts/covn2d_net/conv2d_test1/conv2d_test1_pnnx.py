import os
import numpy as np
import tempfile, zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchvision
except:
    pass

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=2, kernel_size=(3,3), out_channels=4, padding=(1,1), padding_mode='zeros', stride=(2,2))

        archive = zipfile.ZipFile('conv2d_test1.pnnx.bin', 'r')
        self.conv.bias = self.load_pnnx_bin_as_parameter(archive, 'conv.bias', (4), 'float32')
        self.conv.weight = self.load_pnnx_bin_as_parameter(archive, 'conv.weight', (4,2,3,3), 'float32')
        archive.close()

    def load_pnnx_bin_as_parameter(self, archive, key, shape, dtype, requires_grad=True):
        return nn.Parameter(self.load_pnnx_bin_as_tensor(archive, key, shape, dtype), requires_grad)

    def load_pnnx_bin_as_tensor(self, archive, key, shape, dtype):
        _, tmppath = tempfile.mkstemp()
        tmpf = open(tmppath, 'wb')
        with archive.open(key) as keyfile:
            tmpf.write(keyfile.read())
        tmpf.close()
        m = np.memmap(tmppath, dtype=dtype, mode='r', shape=shape).copy()
        os.remove(tmppath)
        return torch.from_numpy(m)

    def forward(self, v_0):
        v_1 = self.conv(v_0)
        return v_1

def export_torchscript():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 2, 5, 5, dtype=torch.float)

    mod = torch.jit.trace(net, v_0)
    mod.save("conv2d_test1_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 2, 5, 5, dtype=torch.float)

    torch.onnx._export(net, v_0, "conv2d_test1_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0'], output_names=['out0'])

def test_inference():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 2, 5, 5, dtype=torch.float)

    return net(v_0)
