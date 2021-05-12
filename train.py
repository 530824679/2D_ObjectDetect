# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : train.py
# Description : train code
# --------------------------------------

import os
import math
import shutil
import numpy as np
import tensorflow as tf

from cfg.config import path_params, model_params, solver_params
from model.loss import Loss
from model.network import Network
from data import dataset, tfrecord
from utils.data_utils import *

def compute_warmup_lr(global_step, warmup_steps, name):
    """

    :param warmup_steps:
    :param name:
    :return:
    """
    with tf.variable_scope(name_or_scope=name):
        warmup_init_learning_rate = solver_params['init_learning_rate'] / 1000.0
        factor = tf.math.pow(solver_params['init_learning_rate'] / warmup_init_learning_rate, 1.0 / warmup_steps)
        warmup_lr = warmup_init_learning_rate * tf.math.pow(factor, global_step)
    return warmup_lr

def train():
    start_step = 0
    input_shape = model_params['input_shape']
    total_epoches = solver_params['total_epoches']
    batch_size = solver_params['batch_size']
    checkpoint_dir = path_params['checkpoints_dir']
    tfrecord_dir = path_params['tfrecord_dir']
    log_dir = path_params['logs_dir']
    initial_weight = path_params['initial_weight']
    restore = solver_params['restore']
    classes = read_class_names(path_params['class_file'])
    class_num = len(classes)

    # 创建相关目录
    ckpt_path = path_params['checkpoints_dir']
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    logs_path = path_params['logs_dir']
    if os.path.exists(logs_path):
        shutil.rmtree(logs_path)
    os.makedirs(logs_path)

    # Set sess configuration
    gpu_options = tf.ConfigProto(allow_soft_placement=True)
    gpu_options.gpu_options.allow_growth = True
    gpu_options.gpu_options.allocator_type = 'BFC'

    # 解析得到训练样本以及标注
    data = tfrecord.TFRecord()
    train_tfrecord = os.path.join(tfrecord_dir, "train.tfrecord")
    data_num = total_sample(train_tfrecord)
    batch_num = int(math.ceil(float(data_num) / batch_size))
    dataset = data.create_dataset(train_tfrecord, batch_num, batch_size=batch_size, is_shuffle=False)

    # 创建训练和验证数据迭代器
    iterator = dataset.make_one_shot_iterator()
    inputs, y_true_20, y_true_40, y_true_80 = iterator.get_next()

    inputs.set_shape([None, input_shape[0], input_shape[1], 3])
    y_true_20.set_shape([None, 20, 20, 3, 5 + class_num])
    y_true_40.set_shape([None, 40, 40, 3, 5 + class_num])
    y_true_80.set_shape([None, 80, 80, 3, 5 + class_num])
    y_true = [y_true_20, y_true_40, y_true_80]

    # 构建网络计算损失
    network = Network(is_train=True)
    logits, preds = network.forward(inputs)

    losses = Loss()
    loss_op = losses.compute_loss(logits, preds, y_true, 'loss')

    #l2_loss = tf.losses.get_regularization_loss()
    #total_loss = loss_op[0] + loss_op[1] + loss_op[2]# + l2_loss

    # define training op
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(solver_params['init_learning_rate'], global_step, solver_params['decay_steps'], solver_params['decay_rate'], staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss_op[0], global_step=global_step)

    # 模型保存
    loader = tf.train.Saver(tf.moving_average_variables())
    save_variable = tf.global_variables()
    saver = tf.train.Saver(save_variable, max_to_keep=1000)

    # 配置tensorboard
    tf.summary.scalar('learn_rate', learning_rate)
    tf.summary.scalar("iou_loss", loss_op[1])
    tf.summary.scalar("conf_loss", loss_op[2])
    tf.summary.scalar("class_loss", loss_op[3])
    tf.summary.scalar('total_loss', loss_op[0])

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph(), flush_secs=60)

    # 开始训练
    with tf.Session(config=gpu_options) as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        if restore == True:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                stem = os.path.basename(ckpt.model_checkpoint_path)
                restore_step = int(stem.split('.')[0].split('-')[-1])
                start_step = restore_step
                sess.run(global_step.assign(restore_step))
                loader.restore(sess, ckpt.model_checkpoint_path)
                print('Restoreing from {}'.format(ckpt.model_checkpoint_path))
            else:
                print("Failed to find a checkpoint")

        summary_writer.add_graph(sess.graph)
        try:
            print('=> Restoring weights from: %s ... ' % initial_weight)
            loader.restore(sess, initial_weight)
        except:
            print('=> %s does not exist !!!' % initial_weight)
            print('=> Now it starts to train from scratch ...')

        print('\n----------- start to train -----------\n')
        for epoch in range(start_step + 1, total_epoches):
            train_epoch_loss, train_epoch_iou_loss, train_epoch_confs_loss, train_epoch_class_loss = [], [], [], []
            for index in tqdm(range(batch_num)):
                _, summary_, loss_, iou_loss_, confs_loss_, class_loss_, global_step_, lr = sess.run(
                    [train_op, summary_op, loss_op[0], loss_op[1], loss_op[2], loss_op[3], global_step, learning_rate])

                train_epoch_loss.append(loss_)
                train_epoch_iou_loss.append(iou_loss_)
                train_epoch_confs_loss.append(confs_loss_)
                train_epoch_class_loss.append(class_loss_)

                summary_writer.add_summary(summary_, global_step_)

            train_epoch_loss, train_epoch_iou_loss, train_epoch_confs_loss, train_epoch_class_loss = np.mean(train_epoch_loss), np.mean(train_epoch_iou_loss), np.mean(train_epoch_confs_loss), np.mean(train_epoch_class_loss)
            print("Epoch: {}, global_step: {}, lr: {:.8f}, total_loss: {:.3f}, iou_loss: {:.3f},confs_loss: {:.3f}, class_loss: {:.3f}".format(epoch, global_step_, lr, train_epoch_loss, train_epoch_iou_loss, train_epoch_confs_loss, train_epoch_class_loss))
            snapshot_model_name = 'tusimple_train_miou={:.4f}.ckpt'.format(train_epoch_iou_loss)
            saver.save(sess, os.path.join(checkpoint_dir, snapshot_model_name), global_step=epoch)

        sess.close()

if __name__ == '__main__':
    train()