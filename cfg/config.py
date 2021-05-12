
import os

path_params = {
    'data_path': "/home/chenwei/HDD/Project/2D_ObjectDetect/datasets/self_datasets/voc",
    'root_path': '/home/chenwei/HDD/Project/2D_ObjectDetect/datasets',
    'class_file': '/home/chenwei/HDD/Project/2D_ObjectDetect/data/classes.txt',
    'train_file': '/home/chenwei/HDD/Project/2D_ObjectDetect/data/train.txt',
    'anchor_file': '/home/chenwei/HDD/Project/2D_ObjectDetect/data/anchors.txt',
    'tfrecord_dir': '/home/chenwei/HDD/Project/2D_ObjectDetect/datasets/tfrecords',
    'logs_dir': './logs',
    'checkpoint_name': '2D_OD',
    'checkpoints_dir': './checkpoints',
    'initial_weight': './weight/model.ckpt'
}


model_params = {
    'input_shape': [640, 640],
    'strides': [8, 16, 32],
    'anchor_per_scale': 3,
    'label_smoothing': 0.01,
    'iou_threshold': 0.5,
    'warm_up_epoch': 3
}

solver_params = {
    'total_epoches': 2000,
    'batch_size': 8,
    'warmup_epoches': 10,
    'init_learning_rate': 0.001,
    'decay_steps': 500,            # 衰变步数
    'decay_rate': 0.95,             # 衰变率
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'restore': False,  # 支持restore
}

test_params = {
    'score_threshold': 0.3,
    'iou_threshold': 0.4
}