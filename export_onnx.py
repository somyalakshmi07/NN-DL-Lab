# export_onnx.py
import torch
from model import CSRNet

model = CSRNet().cuda()
model.load_state_dict(torch.load('checkpoints/csrnet_epoch400.pth'))
model.eval()

dummy = torch.randn(1, 3, 224, 224).cuda()
torch.onnx.export(model, dummy, "csrnet.onnx", opset_version=12)