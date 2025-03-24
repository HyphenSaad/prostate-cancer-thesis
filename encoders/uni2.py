import timm
from torchvision import transforms
import torch

def get_encoder(device):
  model = timm.create_model(
    "hf-hub:MahmoodLab/UNI2-h", 
    pretrained = False,
    img_size = 224,
    patch_size = 14,
    depth = 24,
    num_heads = 24,
    init_values = 1e-5,
    embed_dim = 1536,
    mlp_ratio = 2.66667*2,
    num_classes = 0,
    no_embed_class = True,
    mlp_layer = timm.layers.SwiGLUPacked,
    act_layer = torch.nn.SiLU,
    reg_tokens = 8,
    dynamic_img_size = True,
  )

  model.load_state_dict(torch.load('./prostate-cancer-thesis/encoders/ckpts/uni2.bin', map_location="cpu"), strict=True)
  model.eval()

  return model.to(device)

def custom_transformer():
  transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
  ])
  return transform