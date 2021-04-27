
import os
import json
import glob
import shutil
from tqdm import tqdm
from shutil import copy
import numpy as np

def rename_datas(root_path, dst_path):
    src_image_dir = os.path.join(root_path, "images")
    src_label_dir = os.path.join(root_path, "labels")

    dst_image_dir = os.path.join(dst_path, "images")
    dst_label_dir = os.path.join(dst_path, "labels")

    image_count = 0
    for root, dirs, files in os.walk(src_image_dir):
        for name in files:
            print(os.path.join(root, name))

            src_image_path = os.path.join(root, name)
            src_label_path = os.path.join(root.replace("images", "labels"), name.replace("png", "json"))

            dst_image_path = os.path.join(dst_image_dir, '%08d' % int(image_count))
            dst_label_path = os.path.join(dst_label_dir, '%08d' % int(image_count))

            shutil.copy(src_image_path, dst_image_path + '.png')
            shutil.copy(src_label_path, dst_label_path + '.json')

            image_count += 1

def get_files(path, _ends=['*.json']):
    all_files = []
    for _end in _ends:
        files = glob.glob(os.path.join(path, _end))
        all_files.extend(files)
    file_num = len(all_files)
    return all_files, file_num

def get_path(result_dict):
    obj_path = result_dict['path']
    obj_path = obj_path.split('/')[-1]
    return obj_path

def get_size(result_dict):
    obj_size = result_dict['size']
    width = obj_size["width"]
    height = obj_size["height"]
    return width,height

def get_text_mark(file_path, out_Annotations_path):
    xml_content = []
    with open(file_path, 'r', encoding='utf-8') as fid:
        result_dict = json.load(fid)
        obj = result_dict['outputs']['object']
        file_name = get_path(result_dict)
        name = file_name.split('.')[0]
        width, height = get_size(result_dict)
        xml_content.append("<annotation>")
        xml_content.append("	<folder>" + name+'.jpg' + "</folder>")
        xml_content.append("	<filename>" + file_name + "</filename>")
        xml_content.append("	<size>")
        xml_content.append("		<width>" + str(width) + "</width>")
        xml_content.append("		<height>" + str(height) + "</height>")
        xml_content.append("	</size>")
        xml_content.append("	<segmented>0</segmented>")

        for obj_item in obj:
            cate_name = obj_item['name']
            coords = obj_item['bndbox']
            try:
                bbox = [float(coords['xmin']),float(coords['ymin']),float(coords['xmax']),float(coords['ymax'])]
                xml_content.append("	<object>")
                xml_content.append("		<name>" + cate_name + "</name>")
                xml_content.append("		<pose>Unspecified</pose>")
                xml_content.append("		<truncated>0</truncated>")
                xml_content.append("		<difficult>0</difficult>")
                xml_content.append("		<bndbox>")
                xml_content.append("			<xmin>" + str(int(bbox[0])) + "</xmin>")
                xml_content.append("			<ymin>" + str(int(bbox[1])) + "</ymin>")
                xml_content.append("			<xmax>" + str(int(bbox[2])) + "</xmax>")
                xml_content.append("			<ymax>" + str(int(bbox[3])) + "</ymax>")
                xml_content.append("		</bndbox>")
                xml_content.append("	</object>")
            except:
                continue
        xml_content.append("</annotation>")
        x = xml_content
        xml_content = [x[i] for i in range(0, len(x)) if x[i] != "\n"]
        xml_path = os.path.join(out_Annotations_path, file_name.replace('.png', '.xml'))
        with open(xml_path, 'w+', encoding="utf8") as f:
            f.write('\n'.join(xml_content))
        xml_content[:] = []
        return

def json2voc(src_path, dst_path):
    src_image_dir = os.path.join(src_path, "images")
    src_label_dir = os.path.join(src_path, "labels")

    dst_Annotations_dir = os.path.join(dst_path, "Annotations")
    dst_ImageSets_dir = os.path.join(dst_path, "ImageSets\Main")
    dst_JPEGImages_dir = os.path.join(dst_path, "JPEGImages")

    files, files_len = get_files(src_label_dir)
    bar = tqdm(total=files_len)
    for file in files:
        bar.update(1)
        try:
            get_text_mark(file, dst_Annotations_dir)
        except:
            print(file + 'wrong')
            continue

    filelist = os.listdir(src_image_dir)
    for i in filelist:
        src = os.path.join(os.path.abspath(src_image_dir), i)
        # 重命名
        dst = os.path.join(os.path.abspath(dst_JPEGImages_dir), i.split('.')[0] + '.png')
        # 执行操作
        copy(src, dst)

    bar.close()


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





if __name__ == '__main__':
    json2voc('/home/chenwei/HDD/Project/2D_ObjectDetect/datasets/self_datasets', '/home/chenwei/HDD/Project/2D_ObjectDetect/datasets/self_datasets/voc')
    #rename_datas("/home/chenwei/HDD/Project/datasets/segmentation/周视数据", "/home/chenwei/HDD/Project/2D_ObjectDetect/datasets/self_datasets")