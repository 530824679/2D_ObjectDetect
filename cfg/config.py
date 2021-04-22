
import os

path_params = {
    'data_path': "./data",
    'checkpoints_dir': './checkpoints',

}


model_params = {
    'input_shape': [800, 800],
    'strides': [8, 16, 32],
    'class_num': 5,
}

solver_params = {
    'total_epoches': 2000,
    'batch_size': 8,
    'init_learning_rate': 0.0001,
    'momentum': 0.9,
    'weight_decay': 0.0005
}

test_params = {
    'score_threshold': 0.3,
    'iou_threshold': 0.4
}