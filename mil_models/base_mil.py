import torch
import torch.nn.functional as F
from torch import nn

class BaseMILModel(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.task_adapter = task_logits_adapter

def task_logits_adapter(logits):
  Y_prob = F.softmax(logits, dim = 1)
  Y_hat = torch.topk(logits, 1, dim = 1)[1]
  return logits, Y_prob, Y_hat