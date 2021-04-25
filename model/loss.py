

# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : loss.py
# Description :Yolo_v5 Loss损失函数.
# --------------------------------------

import math
import numpy as np
import tensorflow as tf
from cfg.config import *
from utils.data_utils import *

class Loss(object):
    def __init__(self):
        self.batch_size = solver_params['batch_size']
        self.anchor_per_scale = model_params['anchor_per_scale']
        self.classes = read_class_names(path_params['class_file'])
        self.class_num = len(self.classes)

    def compute_loss(self, pred_conv, pred_bbox, label_bbox, scope='loss'):
        """
        :param pred_conv: [pred_sconv, pred_mconv, pred_lconv]. pred_conv_shape=[batch_size, conv_height, conv_width, anchor_per_scale, 7 + num_classes]
        :param pred_bbox: [pred_sbbox, pred_mbbox, pred_lbbox]. pred_bbox_shape=[batch_size, conv_height, conv_width, anchor_per_scale, 7 + num_classes]
        :param label_bbox: [label_sbbox, label_mbbox, label_lbbox].
        :return:
        """
        with tf.name_scope('smaller_box_loss'):
            loss_sbbox = self.loss_layer(pred_conv[0], pred_bbox[0], label_bbox[0])

        with tf.name_scope('medium_box_loss'):
            loss_mbbox = self.loss_layer(pred_conv[1], pred_bbox[1], label_bbox[1])

        with tf.name_scope('bigger_box_loss'):
            loss_lbbox = self.loss_layer(pred_conv[2], pred_bbox[2], label_bbox[2])

        with tf.name_scope('iou_loss'):
            iou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]

        with tf.name_scope('conf_loss'):
            conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]

        with tf.name_scope('class_loss'):
            class_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]

        return iou_loss, conf_loss, class_loss

    def loss_layer(self, pred_feat, pred_bbox, y_true):
        input_size = stride * output_size
        feature_shape = tf.shape(pred_feat)[1:3]
        predicts = tf.reshape(pred_feat, (-1, feature_shape[0], feature_shape[1], self.anchor_per_scale, 5 + self.class_num))

        conv_raw_conf = predicts[:, :, :, :, 4:5]
        conv_raw_prob = predicts[:, :, :, :, 5:]

        pred_xywh = pred_bbox[:, :, :, :, 0:4]
        pred_conf = pred_bbox[:, :, :, :, 4:5]

        label_xywh = y_true[:, :, :, :, 0:4]
        object_mask = y_true[:, :, :, :, 4:5]
        label_prob = y_true[:, :, :, :, 5:]
        if label_smoothing:
            label_prob = self._label_smoothing(label_prob, label_smoothing)

        iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)
        respond_backgd = (1.0 - respond_bbox) * tf.cast(max_iou < self.iou_loss_thresh, tf.float32)

        input_size = tf.cast(input_size, tf.float32)
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)

        if iou_use == 1:
            diou = tf.expand_dims(self.bbox_diou(pred_xywh, label_xywh), axis=-1)
            iou_loss = respond_bbox * bbox_loss_scale * (1 - diou)
        elif iou_use == 2:
            ciou = tf.expand_dims(self.bbox_ciou(pred_xywh, label_xywh), axis=-1)
            iou_loss = respond_bbox * bbox_loss_scale * (1 - ciou)
        else:
            giou = tf.expand_dims(self.bbox_giou(pred_xywh, label_xywh), axis=-1)
            iou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

        if focal_use:
            focal = self.focal_loss(respond_bbox, pred_conf)
            conf_loss = focal * (respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox,
                                                                                        logits=conv_raw_conf) + \
                                 respond_backgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox,
                                                                                          logits=conv_raw_conf))
            class_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)
        else:
            conf_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox,
                                                                               logits=conv_raw_conf) + \
                        respond_backgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox,
                                                                                 logits=conv_raw_conf)
            class_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

        iou_loss = tf.reduce_mean(tf.reduce_sum(iou_loss, axis=[1, 2, 3, 4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
        class_loss = tf.reduce_mean(tf.reduce_sum(class_loss, axis=[1, 2, 3, 4]))
        return iou_loss, conf_loss, class_loss