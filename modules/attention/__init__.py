import torch
import torch.nn
import torch.nn.functional

class NormalSelfAttention1d(nn.Module):
  def __init__(self, dim=512, dropout_rate=0.0):
    super().__init__()
    self.q_linear = nn.Linear(dim, dim)
    self.k_linear = nn.Linear(dim, dim)
    self.v_linear = nn.Linear(dim, dim)
    self.dropout = nn.Dropout(dropout_rate)
  def forward(self, x):
    x = x.transpose(-1,-2)
    q = self.q_linear(x)
    k = self.k_linear(x)
    v = self.v_linear(x)
    r = self.dropout(F.softmax(q @ k.transpose(-1,-2), dim=-1)) @ v
    return r.transpose(-1,-2)

class AgentSelfAttention1d(nn.Module):
  def __init__(self, dim=512, pool_len=100, dropout_rate=0.0):
    super().__init__()
    self.q_linear = nn.Linear(dim, dim)
    self.k_linear = nn.Linear(dim, dim)
    self.v_linear = nn.Linear(dim, dim)
    self.dropout = nn.Dropout(dropout_rate)
    self.agent_pool = nn.AdaptiveAvgPool1d(agent_size)
  def forward(self, x):
    x = x.transpose(-1,-2)
    q = self.q_linear(x)
    a = self.agent_pool(q.transpose(-1,-2)).transpose(-1,-2)
    k = self.k_linear(x)
    v = self.v_linear(x)
    c = self.dropout(F.softmax(a @ k.transpose(-1, -2), dim=-1)) @ v
    r = self.dropout(F.softmax(q @ a.transpose(-1, -2), dim-1)) @ c
    return r.transpose(-1,-2)

class AnchorSelfAttention1d(nn.Module):
  def __init__(self, dim=512, pool_len=100, dropout_rate=0.0):
    super().__init__()
    self.q_linear = nn.Linear(dim, dim)
    self.k_linear = nn.Linear(dim, dim)
    self.v_linear = nn.Linear(dim, dim)
    self.dropout = nn.Dropout(dropout_rate)
    self.agent_pool = nn.AdaptiveAvgPool1d(agent_size)
  def forward(self, x):
    x = x.transpose(-1,-2)
    q = self.q_linear(x)
    k = self.k_linear(x)
    a = self.agent_pool(k.transpose(-1,-2)).transpose(-1,-2)
    v = self.v_linear(x)
    c = self.dropout(F.softmax(a @ k.transpose(-1, -2), dim=-1)) @ v
    r = self.dropout(F.softmax(q @ a.transpose(-1, -2), dim-1)) @ c
    return r.transpose(-1,-2)
