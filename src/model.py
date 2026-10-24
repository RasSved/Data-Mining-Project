import os
import sys
sys.path.insert(0, os.path.expanduser("~/ai8x/ai8x-training"))
import ai8x
from torch import nn

ai8x.set_device(87, simulate=False, round_avg=False)


class ECGmodel(nn.Module):
    def __init__ (self, num_classes=4, num_channels=1, dimensions=(234,), bias=True):
        super().__init__()

        # Needed 2 pooling layers so we got under 1024 output size
        self.conv1 = ai8x.FusedConv1dReLU(num_channels, 16, 5, padding=2, bias=bias)
        self.conv2 = ai8x.FusedMaxPoolConv1dReLU(16, 16, 5, pool_size=2, pool_stride=2, padding=2, bias=bias)
        self.conv3 = ai8x.FusedMaxPoolConv1dReLU(16, 16, 5, pool_size=2, pool_stride=2, padding=2, bias=bias)

        # 928 = channels * length (what we flatten it to)
        self.fc = ai8x.Linear(928, num_classes, bias=bias, wide=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        print(x.shape)   # (batch, channels, length)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    import torch
    model = ECGmodel()

    # 1. Shape
    out = model(torch.randn(8, 1, 234))
    assert out.shape == (8, 4)

    # 2. Size budget
    n_params = sum(p.numel() for p in model.parameters())
    print("parameters:", n_params)
    assert n_params < 20_000

    # 3. Every layer of the model is an ai8x layer
    for name, m in model.named_children():
        assert isinstance(m, ai8x.QuantizationAwareModule), name