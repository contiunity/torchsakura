import torch

class HookBox:
  def __init__(self):
    self.__box = None
  def set(self, x):
    if isinstance(x, torch.Tensor):
      self.__box = x
    else:
      raise ValueError("Input is not a tensor!")
  def get(self):
    if self.__box == None:
      raise ValueError("Not tensor in box!")
    return self.__box
  def __call__(self, *args):
    ci = self.__box
    if len(args) >= 2:
      raise RuntimeError("Too more args!")
    if len(args) == 1:
      self.set(args[0])
    if ci == None:
      if len(args) == 1:
        return args[0]
      raise ValueError("Not tensor in box!")
    return ci
  def close(self):
    self.__box = None
  def release(self):
    self.__box = None
