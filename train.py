
import os
import torch

def train():

    device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = device.type == "cuda"
    if cuda:

