import os
import json
import xml.etree.ElementTree as ET

import torch

CLASSES = (
    "Person",    # 行人
    "Rider",     # 骑车的人
    "Bike",      # 自行车
    "Motocycle", # 电动车
    "Car",       # 汽车（轿车，SUV）
    "Truck",     # 卡车（货运卡车，水泥车，工程车，清洁车）
    "Bus",       # 巴士（公共汽车，长途客车，面包车）
    "Obstacle"   # 障碍物
)

class Dataset():
    def __init__(self):
        pass

    def load_image(self, index):
