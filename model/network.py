# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : network.py
# Description :YOLO v5 network architecture
# --------------------------------------

import numpy as np
import tensorflow as tf
from cfg.config import *
from model.basenet import *
from utils.data_utils import *

class Network(object):
    def __init__(self, is_train):
        self.is_train = is_train
        self.anchors = read_anchors(path_params['anchor_file'])
        self.classes = read_class_names(path_params['class_file'])
        self.class_num = len(self.classes)
        self.strides = np.array(model_params['strides'])
        self.anchor_per_scale = model_params['anchor_per_scale']

    def forward(self, inputs):
        try:
            conv_lbbox, conv_mbbox, conv_sbbox = self.build_network(inputs)
        except:
            raise NotImplementedError("Can not build up yolov5 network!")

        with tf.variable_scope('pred_sbbox'):
            pred_sbbox = self.reorg_layer(conv_sbbox, self.anchors[0], self.strides[0])

        with tf.variable_scope('pred_mbbox'):
            pred_mbbox = self.reorg_layer(conv_mbbox, self.anchors[1], self.strides[1])

        with tf.variable_scope('pred_lbbox'):
            pred_lbbox = self.reorg_layer(conv_lbbox, self.anchors[2], self.strides[2])

        logits = [conv_sbbox, conv_mbbox, conv_lbbox]
        preds = [pred_sbbox, pred_mbbox, pred_lbbox]
        return logits, preds

    def build_network(self, inputs):

        # backbone
        focus_0 = focus(inputs, 64, 3, 'model/0')
        conv_1 = convBnLeakly(focus_0, 128, 3, 2, "model/1")
        bottleneck_csp_2 = bottleneckCSP(conv_1, 128, 128, 3, True, 0.5, "model/2")
        conv_3 = convBnLeakly(bottleneck_csp_2, 256, 3, 2, 'model/3')
        bottleneck_csp_4 = bottleneckCSP(conv_3, 256, 256, 9, True, 0.5, 'model/4')
        conv_5 = convBnLeakly(bottleneck_csp_4, 512, 3, 2, 'model/5')
        bottleneck_csp_6 = bottleneckCSP(conv_5, 512, 512, 9, True, 0.5, 'model/6')
        conv_7 = convBnLeakly(bottleneck_csp_6, 1024, 3, 2, 'model/7')
        spp_8 = spp(conv_7, 1024, 1024, 5, 9, 13, 'model/8')

        # neck
        bottleneck_csp_9 = bottleneckCSP(spp_8, 1024, 1024, 3, False, 0.5, 'model/9')
        conv_10 = convBnLeakly(bottleneck_csp_9, 512, 1, 1, 'model/10')

        shape = [conv_10.shape[1].value * 2, conv_10.shape[2].value * 2]
        # 0?????????????????????1?????????????????????2????????????????????????3??????????????????
        deconv_11 = tf.image.resize_images(conv_10, shape, method=1)

        concat_12 = tf.concat((deconv_11, bottleneck_csp_6), -1)
        bottleneck_csp_13 = bottleneckCSP(concat_12, 1024, 512, 3, False, 0.5, 'model/13')
        conv_14 = convBnLeakly(bottleneck_csp_13, 256, 1, 1, 'model/14')

        shape = [conv_14.shape[1].value * 2, conv_14.shape[2].value * 2]
        deconv_15 = tf.image.resize_images(conv_14, shape, method=1)

        concat_16 = tf.concat((deconv_15, bottleneck_csp_4), -1)
        bottleneck_csp_17 = bottleneckCSP(concat_16, 512, 256, 3, False, 0.5, 'model/17')
        conv_18 = convBnLeakly(bottleneck_csp_17, 256, 3, 2, 'model/18')

        concat_19 = tf.concat((conv_18, conv_14), -1)
        bottleneck_csp_20 = bottleneckCSP(concat_19, 512, 512, 3, False, 0.5, 'model/20')
        conv_21 = convBnLeakly(bottleneck_csp_20, 512, 3, 2, 'model/21')

        concat_22 = tf.concat((conv_21, conv_10), -1)
        bottleneck_csp_23 = bottleneckCSP(concat_22, 1024, 1024, 3, False, 0.5, 'model/23')

        # head
        conv_24_m0 = conv(bottleneck_csp_17, 3 * (self.class_num + 5), 1, 1, 'model/24/m/0', add_bias=True)
        conv_24_m1 = conv(bottleneck_csp_20, 3 * (self.class_num + 5), 1, 1, 'model/24/m/1', add_bias=True)
        conv_24_m2 = conv(bottleneck_csp_23, 3 * (self.class_num + 5), 1, 1, 'model/24/m/2', add_bias=True)

        return conv_24_m0, conv_24_m1, conv_24_m2

    def reorg_layer(self, feature_maps, anchors, strides):
        """
        ??????????????????????????????
        :param feature_maps:????????????????????????
        :param anchors:??????????????????anchor??????
        :param stride:????????????????????????????????????
        :return: ???????????????????????? shape=[batch_size, feature_size, feature_size, anchor_per_scale, 5 + class_num]
        """
        feature_shape = tf.shape(feature_maps)[1:3]
        batch_size = tf.shape(feature_maps)[0]
        anchor_per_scale = len(anchors)

        predict = tf.reshape(feature_maps, (batch_size, feature_shape[0], feature_shape[1], anchor_per_scale, 5 + self.class_num))
        conv_raw_xy = predict[:, :, :, :, 0:2]
        conv_raw_wh = predict[:, :, :, :, 2:4]
        conv_raw_conf = predict[:, :, :, :, 4:5]
        conv_raw_prob = predict[:, :, :, :, 5:]

        y = tf.tile(tf.range(feature_shape[0], dtype=tf.int32)[:, tf.newaxis], [1, feature_shape[0]])
        x = tf.tile(tf.range(feature_shape[1], dtype=tf.int32)[tf.newaxis, :], [feature_shape[1], 1])

        xy_cell = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_cell = tf.tile(xy_cell[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
        xy_cell = tf.cast(xy_cell, tf.float32)

        bboxes_xy = (tf.sigmoid(conv_raw_xy) + xy_cell) * strides
        bboxes_wh = (tf.sigmoid(conv_raw_wh) * anchors) * strides

        pred_xywh = tf.concat([bboxes_xy, bboxes_wh], axis=-1)
        pred_box_confidence = tf.sigmoid(conv_raw_conf)
        pred_box_class_prob = tf.sigmoid(conv_raw_prob)
        return tf.concat([pred_xywh, pred_box_confidence, pred_box_class_prob], axis=-1)