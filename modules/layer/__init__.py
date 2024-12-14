import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1dLayer(nn.Conv1d):
  def __init__(self, in_channels, out_channels, kernel_size, *args, **kwargs):
    super().__init__(in_channels, out_channels * 2, kernel_size, *arg, **kwargs)
    self.channels = out_channels
    nn.init.kaiming_normal_(self.weight)
  def forward(self, x):
    result = super().forward(x)
    result, gate = torch.split(result, [self.channels, self.channels], dim=-2)
    return result * torch.sigmoid(gate)

class Conv2dLayer(nn.Conv2d):
  def __init__(self, in_channels, out_channels, kernel_size, *args, **kwargs):
    super().__init__(in_channels, out_channels * 2, kernel_size, *arg, **kwargs)
    self.channels = out_channels
    nn.init.kaiming_normal_(self.weight)
  def forward(self, x):
    result = super().forward(x)
    result, gate = torch.split(result, [self.channels, self.channels], dim=-3)
    return result * torch.sigmoid(gate)

class Pointwise1dLayer(SDKConv1dLayer):
  def __init__(self, in_channels, out_channels):
    super().__init__(in_channels, out_channels, 1)

class Pointwise2dLayer(SDKConv2dLayer):
  def __init__(self, in_channels, out_channels):
    super().__init__(in_channels, out_channels, 1)
