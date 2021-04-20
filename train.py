
import os
import torch

from utils.gpu import *

def train():

    device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = device.type == "cuda"
    if cuda:
        get_gpu_prop()
    print("\ndevice: {}".format(device))

    # Dataset Load
    dataset_train = yolo.datasets(args.dataset, file_roots[0], ann_files[0], train=True)
    dataset_test = yolo.datasets(args.dataset, file_roots[1], ann_files[1], train=True)  # set train=True for eval