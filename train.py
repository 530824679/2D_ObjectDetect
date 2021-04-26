# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : train.py
# Description : train code
# --------------------------------------

import os
import shutil
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from cfg.config import path_params, model_params, solver_params
from model.loss import Loss
from model.network import Network
from data import dataset, tfrecord


def train():
    input_height = model_params['input_height']
    input_width = model_params['input_width']
    total_epoches = solver_params['total_epoches']
    warm_up_epoch = solver_params['warm_up_epoch']
    warm_up_lr = solver_params['warm_up_lr']
    val_step = solver_params['val_step']
    log_step = solver_params['log_step']
    display_step = solver_params['display_step']
    batch_size = solver_params['batch_size']
    checkpoint_dir = path_params['checkpoints_dir']
    checkpoints_name = path_params['checkpoints_name']
    tfrecord_dir = path_params['tfrecord_dir']
    train_tfrecord_name = path_params['train_tfrecord_name']
    val_tfrecord_name = path_params['val_tfrecord_name']
    log_dir = path_params['logs_dir']

    initial_weight = path_params['initial_weight']
    first_stage_epochs = solver_params['first_stage_epochs']
    second_stage_epochs = solver_params['second_stage_epochs']
    warmup_periods = solver_params['warmup_epoches']
    moving_ave_decay =
    max_bbox_per_scale = 150

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
    with tf.name_scope('input'):
        data = tfrecord.TFRecord()
        train_tfrecord = os.path.join(tfrecord_dir, train_tfrecord_name)
        val_tfrecord = os.path.join(tfrecord_dir, val_tfrecord_name)
        train_dataset = data.create_dataset(train_tfrecord, batch_size=batch_size, is_shuffle=False, n_repeats=total_epoches)
        val_dataset = data.create_dataset(val_tfrecord, batch_size=batch_size, is_shuffle=False, n_repeats=-1)

        # 创建训练和验证数据迭代器
        train_iterator = train_dataset.make_one_shot_iterator()
        val_iterator = val_dataset.make_one_shot_iterator()

        train_images, y_true_19, y_true_38, y_true_76 = train_iterator.get_next()
        y_true = [y_true_19, y_true_38, y_true_76]

        # tf.data pipeline will lose the data shape, so we need to set it manually
        train_images.set_shape([None, input_height, input_width, 1])
        y_true_19.set_shape([None, 19, 19, 2, 7 + len(model_params['classes'])])
        y_true_38.set_shape([None, 38, 38, 2, 7 + len(model_params['classes'])])
        y_true_76.set_shape([None, 76, 76, 2, 7 + len(model_params['classes'])])
        for y in y_true:
            y.set_shape([None, None, None, None, None])

    # 构建网络计算损失
    with tf.name_scope('define_loss'):
        network = Network(is_train=True)
        logits, preds = network.forward(train_images)

        losses = Loss()
        loss_op = losses.compute_loss(logits, preds, y_true, 'loss')

        vars = tf.trainable_variables()
        l2_reg_loss_op = tf.add_n([tf.nn.l2_loss(var) for var in vars]) * solver_params['weight_decay']
        total_loss = loss_op[0] + loss_op[1] + loss_op[2] + l2_reg_loss_op

    # 创建全局的步骤
    with tf.name_scope('learn_rate'):
        global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES], name='global_step')
        warmup_steps = tf.constant(warmup_periods * steps_per_period, dtype=tf.float64, name='warmup_steps')
        train_steps = tf.constant((first_stage_epochs + second_stage_epochs) * steps_per_period, dtype=tf.float64, name='train_steps')

        learn_rate = tf.cond(pred=global_step < warmup_steps, true_fn=lambda: global_step / warmup_steps * learn_rate_init, false_fn=lambda: learn_rate_end + 0.5 * (learn_rate_init - learn_rate_end) * (1 + tf.cos((global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi)))
        global_step_update = tf.assign_add(global_step, 1.0)

    # 设置权重衰减
    with tf.name_scope('define_weight_decay'):
        moving_ave = tf.train.ExponentialMovingAverage(moving_ave_decay).apply(tf.trainable_variables())

    # 第一阶段优化器
    with tf.name_scope('define_first_stage_train'):
        first_stage_trainable_var_list = []
        for var in tf.trainable_variables():
            var_name = var.op.name
            var_name_mess = str(var_name).split('/')
            bboxes = ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']

            if var_name_mess[0] in bboxes:
                first_stage_trainable_var_list.append(var)

        first_stage_optimizer = tf.train.AdamOptimizer(learn_rate).minimize(total_loss, var_list=first_stage_trainable_var_list)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            with tf.control_dependencies([first_stage_optimizer, global_step_update]):
                with tf.control_dependencies([moving_ave]):
                    train_op_with_frozen_variables = tf.no_op()

    # 第二阶段优化器
    with tf.name_scope('define_second_stage_train'):
        second_stage_trainable_var_list = tf.trainable_variables()
        second_stage_optimizer = tf.train.AdamOptimizer(learn_rate).minimize(total_loss, var_list=second_stage_trainable_var_list)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            with tf.control_dependencies([second_stage_optimizer, global_step_update]):
                with tf.control_dependencies([moving_ave]):
                    train_op_with_all_variables = tf.no_op()

    # 模型保存
    with tf.name_scope('loader_and_saver'):
        loader = tf.train.Saver(tf.moving_average_variables())
        save_variable = tf.global_variables()
        saver = tf.train.Saver(save_variable, max_to_keep=1000)

    # 配置tensorboard
    with tf.name_scope('summary'):
        tf.summary.scalar('learn_rate', learn_rate)
        tf.summary.scalar("iou_loss", loss_op[0])
        tf.summary.scalar("conf_loss", loss_op[1])
        tf.summary.scalar("class_loss", loss_op[2])
        tf.summary.scalar('total_loss', total_loss)

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph(), flush_secs=60)

    # 开始训练
    with tf.Session(config=gpu_options) as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        summary_writer.add_graph(sess.graph)

        try:
            print('=> Restoring weights from: %s ... ' % initial_weight)
            loader.restore(sess, initial_weight)
        except:
            print('=> %s does not exist !!!' % initial_weight)
            print('=> Now it starts to train from scratch ...')
            first_stage_epochs = 0

        saving = 0.0
        for epoch in range(1, (1 + first_stage_epochs + second_stage_epochs)):
            if epoch <= first_stage_epochs:
                train_op = train_op_with_frozen_variables
            else:
                train_op = train_op_with_all_variables

            pbar = tqdm(trainset)
            train_epoch_loss = []

            for train_data in pbar:
                _, summary, train_step_loss, global_step_val = sess.run(
                    [train_op, write_op, loss, global_step],
                    feed_dict={input_data: train_data[0],
                               label_sbbox: train_data[1], label_mbbox: train_data[2],
                               label_lbbox: train_data[3],
                               true_sbboxes: train_data[4], true_mbboxes: train_data[5],
                               true_lbboxes: train_data[6],
                               trainable: True, })

                train_epoch_loss.append(train_step_loss)
                summary_writer.add_summary(summary, global_step_val)
                pbar.set_description('train loss: %.2f' % train_step_loss)

            train_epoch_loss = np.mean(train_epoch_loss)
            train_epoch_loss = np.mean(train_epoch_loss)

            ckpt_file = os.path.join(ckpt_path, 'social_test-loss=%.4f.ckpt' % (test_epoch_loss))
            if saving == 0.0:
                saving = train_epoch_loss
                print('=> Epoch: %2d Train loss: %.2f' % (epoch, train_epoch_loss))

            elif saving > train_epoch_loss:
                print('=> Epoch: %2d Train loss: %.2f Saving %s ...' %(epoch, train_epoch_loss, ckpt_file))
                saver.save(sess, ckpt_file, global_step=epoch)
                saving = train_epoch_loss

            else:
                print('=> Epoch: %2d Train loss: %.2f' % (epoch, train_epoch_loss))





        print('\n----------- start to train -----------\n')
        for epoch in range(total_epoches):
            _, summary, loss_, global_step_, lr = sess.run([train_op, summary_op, loss_op, global_step, learning_rate])
            summary_writer.add_summary(summary, global_step=global_step_)

            print("Epoch: {}, global_step: {}, lr: {:.8f}, total_loss: {:.3f}, loss_ciou: {:.3f}, loss_reim: {:.3f}, loss_conf: {:.3f}, loss_class: {:.3f}, recall50: {:.3f}, recall75: {:.3f}, avg_iou: {:.3f}".format(
                    epoch, global_step_, lr, loss_[0], loss_[1], loss_[2], loss_[3], loss_[4], loss_[5], loss_[6], loss_[7]))

            if epoch % solver_params['save_step'] == 0 and epoch > 0:
                save_path = saver.save(sess, os.path.join(checkpoint_dir, checkpoints_name), global_step=epoch)
                print('Save modle into {}....'.format(save_path))

            if epoch % log_step == 0 and epoch > 0:
                summary_writer.add_summary(summary, global_step=epoch)




        sess.close()

if __name__ == '__main__':
    train()