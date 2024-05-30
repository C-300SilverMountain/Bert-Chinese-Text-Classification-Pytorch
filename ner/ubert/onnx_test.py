
# import onnx
# onnx_mdel = onnx.load("onnx/ubert.onnx")
# for node in onnx_model.graph.node:
#     print(node)
# import torch
# import torchvision.models as models
# from torchinfo import summary
# resnet18 = torch.load('checkpoint/last.ckpt') # 实例化模型
# summary(resnet18, (1, 3, 224, 224))

import torch

params = torch.load('checkpoint/last.ckpt')
for key, value in params.items():
    print(f"Layer Name: {key}")
    print(f"Shape: {value}")
