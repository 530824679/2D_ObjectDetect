
import torch
import torch.nn as nn

class YOLOv5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        