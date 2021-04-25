# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : train.py
# Description : train code
# --------------------------------------

import os
import tensorflow as tf
from cfg.config import *
from model.network import Network
from data import dataset, tfrecord

def train():

    # Set sess configuration
    gpu_options = tf.ConfigProto(allow_soft_placement=True)
    gpu_options.gpu_options.allow_growth = True
    gpu_options.gpu_options.allocator_type = 'BFC'
    sess = tf.Session(config=gpu_options)

    # 解析得到训练样本以及标注
    train_dataset = DatasetFeeder(flags='train')
    steps_per_epoch = len(train_dataset)

    # define graph input tensor
    with tf.variable_scope(name_or_scope='graph_input_node'):
        input_src_image, input_binary_label_image, input_instance_label_image = train_dataset.next_batch(batch_size=solver_params['batch_size'])
