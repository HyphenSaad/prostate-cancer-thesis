import torch

from constants.mil_models import MILModels

def find_mil_model(
  model_name: str,
  in_dim: int,
  n_classes: int,
  drop_out: float,
):
  if model_name.lower() == MILModels.TRANS_MIL.value.lower():
    from .trans_mil import TransMIL
    model = TransMIL(in_dim, n_classes, drop_out = drop_out, activation = 'relu')
    return model
  else:
    raise NotImplementedError(f'MIL model {model_name} not implemented!')