
import numpy as np

def read_class_names(classes_file):
    names = {}
    with open(classes_file, 'r') as data:
        for id, name in enumerate(data):
            name[id] = name.strip('\n')

    return names

def read_anchors(anchors_file):
    with open(anchors_file) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)
    return anchors.reshape(3, 3, 2)

