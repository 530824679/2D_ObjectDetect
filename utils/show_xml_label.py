import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)

import os
import cv2
import re

pattens = ['name', 'xmin', 'ymin', 'xmax', 'ymax']

def get_annotations(xml_path):
    bbox = []
    with open(xml_path, 'r') as f:
        text = f.read().replace('\n', 'return')
        p1 = re.compile(r'(?<=<object>)(.*?)(?=</object>)')
        result = p1.findall(text)
        for obj in result:
            tmp = []
            for patten in pattens:
                p = re.compile(r'(?<=<{}>)(.*?)(?=</{}>)'.format(patten, patten))
                if patten == 'name':
                    tmp.append(p.findall(obj)[0])
                else:
                    tmp.append(int(float(p.findall(obj)[0])))
            bbox.append(tmp)
    return bbox

def save_viz_image(image_path, xml_path, save_path):
    bbox = get_annotations(xml_path)
    image = cv2.imread(image_path)
    for info in bbox:
        cv2.rectangle(image, (info[1], info[2]), (info[3], info[4]), (255, 0, 0), thickness=2)
        cv2.putText(image, info[0], (info[1], info[2]), cv2.FONT_HERSHEY_PLAIN, 2,(0, 0, 255), 2)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    cv2.imwrite(os.path.join(save_path, image_path.split('/')[-1]), image)

def show_and_save(path):
    image_dir = 'JPEGImages'
    xml_dir = 'Annotations'
    save_dir = 'viz_images'
    image_path1 = os.path.join(os.path.abspath(path), image_dir)
    xml_path1 = os.path.join(os.path.abspath(path), xml_dir)
    save_path = os.path.join(os.path.abspath(path), save_dir)
    image_list = os.listdir(image_path1)
    for i in image_list:
        image_path = os.path.join(image_path1, i)
        xml_path = os.path.join(xml_path1, i.replace('.jpg', '.xml'))
        save_viz_image(image_path, xml_path, save_path)



if __name__ == '__main__':
    path = '/home/chenwei/HDD/Project/2D_ObjectDetect/datasets/self_datasets/voc'
    show_and_save(path)