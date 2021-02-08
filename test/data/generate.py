# This generates models for the onnx model optimization tests

import torch
from torch.cuda import random
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
        "name": "conv_no_bias_batchnorm",
        "model": nn.Sequential(
            nn.Conv2d(3, 4, (3,3), bias=False),
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

batch_inputs = [
    torch.rand([10,3,6,6]),
    torch.rand([10,3,6,6]),
    torch.rand([10,3,6,6]),
    torch.rand([10,3,6,6])
]

dummy_inputs = [
    torch.arange(0,1*3*6*6, dtype=torch.float32).reshape((1,3,6,6)),
    torch.arange(0,1*3*6*6, dtype=torch.float32).reshape((1,3,6,6)),
    torch.arange(0,1*3*6*6, dtype=torch.float32).reshape((1,3,6,6)),
    torch.arange(0,1*3*6*6, dtype=torch.float32).reshape((1,3,6,6))
]

iterations = 100

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif type(m) == nn.Conv2d:
        print("init conv2d")
        torch.nn.init.xavier_uniform_(m.weight)

for i in range(len(models)):
    model = models[i]

    model["model"].apply(init_weights)

    for j in range(iterations):
        res = model["model"](batch_inputs[i])
        loss = (res*res).sum()
        print(loss)
        loss.backward()

    torch.onnx.export(model["model"],
                      dummy_inputs[i],
                      f"test/data/onnx/optimizations/{model['name']}.onnx",
                      training=torch.onnx.TrainingMode.TRAINING)
