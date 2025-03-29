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
  elif model_name.lower() == MILModels.DS_MIL.value.lower():
    from .ds_mil import FCLayer, BClassifier, MILNet
    dropout = 0.25 if drop_out else 0
    i_classifier = FCLayer(in_size = in_dim, out_size = n_classes)
    b_classifier = BClassifier(input_size = in_dim, output_class = n_classes, dropout_v = dropout)
    model = MILNet(i_classifier, b_classifier)
    return model
  else:
    raise NotImplementedError(f'MIL model {model_name} not implemented!')