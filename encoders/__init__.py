import torch
import torchvision

from constants.encoders import Encoders

def get_encoder(
  encoder_name: str,
  device: torch.device,
  gpu_num: int
) -> torch.nn.Module:
  encoder = None
  
  if encoder_name in [
    Encoders.RESNET18.value,
    Encoders.RESNET34.value,
    Encoders.RESNET50.value,
    Encoders.RESNET101.value,
    Encoders.RESNET152.value,
  ]:
    from encoders.resnet import get_encoder_baseline
    encoder = get_encoder_baseline(encoder_name, pretrained=True).to(device)
  else:
    raise NotImplementedError(f'Encoder \'{encoder_name}\' is not implemented, yet!')

  if encoder_name in [
    Encoders.RESNET18.value,
    Encoders.RESNET34.value,
    Encoders.RESNET50.value,
    Encoders.RESNET101.value,
    Encoders.RESNET152.value,
  ]:
    if gpu_num > 1: encoder = torch.nn.parallel.DataParallel(encoder)
    encoder = encoder.eval()

  return encoder

def get_custom_transformer(encoder_name: str) -> torchvision.transforms:
  transformer = None
  
  if encoder_name in [
    Encoders.RESNET18.value,
    Encoders.RESNET34.value,
    Encoders.RESNET50.value,
    Encoders.RESNET101.value,
    Encoders.RESNET152.value,
  ]:
    from encoders.resnet import custom_transformer
    transformer = custom_transformer()
  else:
    raise NotImplementedError(f'Transformer for \'{encoder_name}\' is not implemented, yet!')

  return transformer