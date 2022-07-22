import torch
from torch import nn
from torchvision.models import mobilenet_v2
import torch, torchvision.models

model = torchvision.models.vgg16()

path = 'number_detectionv8.pth'
map_location = lambda storage, loc: storage
if torch.cuda.is_available():
    map_location = None
model.load_state_dict(torch.load(path, map_location=map_location))

img_size = (800, 800)
batch_size = 1
onnx_model_path = 'model.onnx'

sample_input = torch.rand((batch_size, 3, *img_size))

y = model(sample_input)

torch.onnx.export(
    model,
    sample_input,
    onnx_model_path,
    verbose=False,
    input_names=['input'],
    output_names=['output'],
    opset_version=12
)
