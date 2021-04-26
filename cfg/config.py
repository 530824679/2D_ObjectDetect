
import os

path_params = {
    'data_path': "/home/chenwei/HDD/Project/2D_ObjectDetect/datasets/self_datasets",
    'root_path': '/home/chenwei/HDD/Project/2D_ObjectDetect/datasets',
    'class_file': '/home/chenwei/HDD/Project/2D_ObjectDetect/data/classes.txt',
    'train_file': '/home/chenwei/HDD/Project/2D_ObjectDetect/data/train.txt',
    'anchor_file': '/home/chenwei/HDD/Project/2D_ObjectDetect/data/anchors.txt',
    'logs_dir': './logs',
    'checkpoint_name': '2D_OD',
    'checkpoints_dir': './checkpoints',
    'initial_weight': './weight/model.ckpt'
}


model_params = {
    'input_shape': [800, 800],
    'strides': [8, 16, 32],
    'anchor_per_scale': 3,
    'label_smoothing': 0.01,
    'iou_threshold': 0.5
}

solver_params = {
    'total_epoches': 2000,
    'batch_size': 8,
    'warmup_epoches': 10,
    'first_stage_epochs': 100,
    'second_stage_epochs': 1000,
    'init_learning_rate': 0.0001,
    'momentum': 0.9,
    'weight_decay': 0.0005
}

test_params = {
    'score_threshold': 0.3,
    'iou_threshold': 0.4
}