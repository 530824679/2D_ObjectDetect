
import tensorflow as tf
from cfg.config import *

def post_process(inputs, grids, strides, anchors, class_num):

    total = []
    for i, logits in enumerate(inputs):

        logits_xy = (logits[..., :2] * 2. - 0.5 + grids[i]) * strides[i]
        logits_wh = ((logits[..., 2:4] * 2) ** 2) * anchors[i]
        logits_new = tf.concat((logits_xy, logits_wh, logits[..., 4:]), axis=-1)

    # 过滤低置信度的目标
    mask = total[:, 4] > 0.15
    total = tf.boolean_mask(total, mask)

    # x,y,w,h ——> x1,y1,x2,y2
    x, y, w, h, conf, prob = tf.split(total, [1, 1, 1, 1, 1, class_num], axis=-1)
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x1 + w
    y2 = y1 + h

    conf_prob = conf * prob
    scores = tf.reduce_max(conf_prob, axis=-1)
    labels = tf.cast(tf.argmax(conf_prob, axis=-1), tf.float32)
    boxes = tf.concat([x1, y1, x2, y2], axis=1)

    indices = tf.image.non_max_suppression(boxes, scores, max_output_size=1000, iou_threshold=test_params['iou_threshold'], score_threshold=test_params['score_threshold'])

    boxes = tf.gather(boxes, indices)
    scores = tf.reshape(tf.gather(scores, indices), [-1, 1])
    labels = tf.reshape(tf.gather(labels, indices), [-1, 1])

    output = tf.concat([boxes, scores, labels], axis=-1)
    return output
