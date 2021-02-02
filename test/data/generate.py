import torch
import torch.nn as nn

models = [
    {
        "name": "conv_batchnorm",
        "model": nn.Sequential(
            nn.Conv2d(3, 4, (3,3)),
            nn.BatchNorm2d(4)
        )
    },
    {
        "name": "conv_batchnorm_relu",
        "model": nn.Sequential(
            nn.Conv2d(3, 4, (3,3)),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )
    },
    {
        "name": "conv_batchnorm_relu6",
        "model": nn.Sequential(
            nn.Conv2d(3, 4, (3,3)),
            nn.BatchNorm2d(4),
            nn.ReLU6()
        )
    }
]

inputs = [
    torch.arange(0,1*3*6*6, dtype=torch.float32).reshape((1,3,6,6)),
    torch.arange(0,1*3*6*6, dtype=torch.float32).reshape((1,3,6,6)),
    torch.arange(0,1*3*6*6, dtype=torch.float32).reshape((1,3,6,6))
]

for i in range(len(models)):
    model = models[i]
    res = model["model"](inputs[i])

    torch.onnx.export(model["model"], inputs[i], f"test/data/onnx/optimizations/{model['name']}.onnx")
