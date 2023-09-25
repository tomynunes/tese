import os
import argparse
from datetime import datetime
import json
import yaml
import tqdm
from PIL import Image
import timm
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
 
import wandb
 
class Model(nn.Module):
    """Example of a model using PyTorch
 
    This specific model is a pretrained ResNet18 model from torch hub
    with a custom head.
 
    """
 
    def __init__(self):
        super().__init__()
 
        self.base = timm.create_model('resnet50', pretrained=True)
        self.base.fc = nn.Linear(1000, 2)
 
    def forward(self, x):
        x = self.base(x)

        return x
 


model = Model()
print(model)