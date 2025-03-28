import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision import transforms

from constants.encoders import Encoders

model_urls = {
  Encoders.RESNET18.value: 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
  Encoders.RESNET34.value: 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
  Encoders.RESNET50.value: 'http://download.pytorch.org/models/resnet50-19c8e357.pth',
  Encoders.RESNET101.value: 'http://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
  Encoders.RESNET152.value: 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class BasicBlockBaseline(nn.Module):
  expansion = 1

  def __init__(
    self,
    inplanes,
    planes,
    stride = 1,
    downsample = None
  ):
    super(BasicBlockBaseline, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out

class BottleneckBaseline(nn.Module):
  expansion = 4

  def __init__(
    self,
    inplanes,
    planes,
    stride = 1,
    downsample = None
  ):
    super(BottleneckBaseline, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * self.expansion)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out

class ResNetBaseline(nn.Module):
  def __init__(self, block, layers):
    self.inplanes = 64
    super(ResNetBaseline, self).__init__()
    
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    # self.layer4 = self._make_layer(block, 512, layers[3], stride=2) # FIX: UPDATED_OTHER_THAN_GPFM_TOOLKIT_CODE
    self.avgpool = nn.AdaptiveAvgPool2d(1) 

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    # x = self.layer4(x) # FIX: UPDATED_OTHER_THAN_GPFM_TOOLKIT_CODE

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)

    return x

def get_encoder_baseline(encoder_name, pretrained = False):
  encoder = None

  if encoder_name == Encoders.RESNET18.value: 
    encoder = ResNetBaseline(BasicBlockBaseline, [2, 2, 2, 2])
  elif encoder_name == Encoders.RESNET34.value:
    encoder = ResNetBaseline(BasicBlockBaseline, [3, 4, 6, 3])
  elif encoder_name == Encoders.RESNET50.value:
    encoder = ResNetBaseline(BottleneckBaseline, [3, 4, 6, 3])
  elif encoder_name == Encoders.RESNET101.value:
    encoder = ResNetBaseline(BottleneckBaseline, [3, 4, 23, 3])
  elif encoder_name == Encoders.RESNET152.value:
    encoder = ResNetBaseline(BottleneckBaseline, [3, 8, 36, 3])

  if pretrained:
    encoder = load_pretrained_weights(encoder, encoder_name)

  return encoder

def load_pretrained_weights(encoder, name):
  pretrained_dict = model_zoo.load_url(model_urls[name])
  encoder.load_state_dict(pretrained_dict, strict=False)
  return encoder

def custom_transformer():
  transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
      mean = (0.485, 0.456, 0.406),
      std = (0.229, 0.224, 0.225)
    ),
  ])
  return transform