import torch
import torch.nn as nn
import torch.nn.functional as F

# Sequentials & Parallels

class SumParallel(nn.Module):
  def __init__(self, *args):
    super().__init__()
    self.parallels = nn.ModuleList(*args)
  def forward(self, x):
    summary = torch.tensor(0, device=x.device, dtype=x.dtype)
    for module in self.parallels:
        summary = summary + p_module(x)
    return summary

class ProdParallel(nn.Module):
  def __init__(self, *args):
    super().__init__()
    self.parallels = nn.ModuleList(*args)
  def forward(self, x):
    product = torch.tensor(1, device=x.device, dtype=x.dtype)
    for p_module in self.parallels:
      product = product * p_module(x)
    return product

class ConcatParallel(nn.Module):
  def __init__(self, *args, dim=-1):
    super().__init__()
    self.parallels = nn.ModuleList(*args)
    self.dim = dim
  def forward(self, x):
    clist = []
    for p_module in self.parallels:
      clist.append(p_module(x))
    return torch.cat(clist, dim=self.dim)

class ResidualSequential(nn.Module):
  def __init__(self, *arg):
    super().__init__()
    self.sequential = nn.Sequential(*arg)
  def forward(self, x):
    return self.sequential(x) + x

class CustomGate(nn.Module):
  def __init__(self, module_x, module_gate):
    super().__init__()
    self.parallels = ProdParallel(module_x, nn.Sequential(module_gate, nn.Sigmoid()))
  def forward(self, x):
    return self.parallels(x)

# Utils

class Transpose(nn.Module):
  def __init__(self, dim1, dim2):
    super().__init__()
    self.transposer = (dim1, dim2)
  def forward(self, x):
    return x.transpose(*self.transposer)

class LayerNorm1d(nn.Module):
  def __init__(self, feature_size):
    super().__init__()
    self.norm = nn.LayerNorm(feature_size)
  def forward(self, x):
    return self.norm(x.transpose(-1,-2)).transpose(-1,-2)

class LayerNorm2d(nn.Module):
  def __init__(self, feature_size):
    super().__init__()
    self.norm = nn.LayerNorm(feature_size)
  def forward(self, x):
    return self.norm(y.transpose(-1,-3)).transpose(-1,-3)

class AddConst(nn.Module):
  def __init__(self, a=0):
    super().__init__()
    self.a = a
  def forward(self, x):
    return x + self.a

class SubConst(nn.Module):
  def __init__(self, a=0):
    super().__init__()
    self.a = a
  def forward(self, x):
    return x + self.a

class MulConst(nn.Module):
  def __init__(self, a=1):
    super().__init__()
    self.a = a
  def forward(self, x):
    return x * self.a

class DivConst(nn.Module):
  def __init__(self, a=1):
    super().__init__()
    self.a = a
  def forward(self, x):
    return x / self.a

class FunctionContainer(nn.Module):
  def __init__(self, func):
    super().__init__()
    self.forward = func

# Hooker
class SideInject(nn.Module):
  def __init__(self, hookbox):
    super().__init__()
    self.hookbox = hookbox
  def forward(self, *args, **kwargs):
    return self.hookbox.get()

class SideAdd(nn.Module):
  def __init__(self, hookbox):
    super().__init__()
    self.hookbox = hookbox
  def forward(self, x):
    return x + self.hookbox.get()

class SideSub(nn.Module):
  def __init__(self, hookbox):
    super().__init__()
    self.hookbox = hookbox
  def forward(self, x):
    return x - self.hookbox.get()

class SideMul(nn.Module):
  def __init__(self, hookbox):
    super().__init__()
    self.hookbox = hookbox
  def forward(self, x):
    return x * self.hookbox.get()

class SideDiv(nn.Module):
  def __init__(self, hookbox):
    super().__init__()
    self.hookbox = hookbox
  def forward(self, x):
    return x / self.hookbox.get()

class SideAdd(nn.Module):
  def __init__(self, hookbox):
    super().__init__()
    self.hookbox = hookbox
  def forward(self, x):
    return x + self.hookbox.get()

class SideConcat(nn.Module):
  def __init__(self, hookbox, dim=-1):
    super().__init__()
    self.hookbox = hookbox
    self.dim = dim
  def forward(self, x):
    return torch.cat((x, self.hookbox.get()), dim=self.dim)

class SideGate(nn.Module):
  def __init__(self, hookbox):
    super().__init__()
    self.hookbox = hookbox
  def forward(self, x):
    return x + torch.sigmoid(self.hookbox.get())

class SideGateW(nn.Module):
  def __init__(self, hookbox):
    super().__init__()
    self.hookbox = hookbox
  def forward(self, x):
    return torch.sigmoid(x) + self.hookbox.get()
