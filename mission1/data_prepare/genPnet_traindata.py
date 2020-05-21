# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : Tam KaHou
import sys
import cv2
import random
import os
sys.path.append(os.getcwd())
import numpy as np
from utilize import *
from imutils import paths
import xml.etree.ElementTree as ET
import argparse
#生成Pnet的训练集和预测集们，根据不同的任务产生不同IOU的数据

parser = argparse.ArgumentParser(description='Generate Pnet_traindata')
parser.add_argument("--train_image_dir", dest='train_img_dir', help=
    "train image path", default="../Plate_dataset/AC/train/jpeg/", type=str)
parser.add_argument("--test_image_dir", dest='test_img_dir', help=
    "test image path", default="../Plate_dataset/AC/test/jpeg/", type=str)
parser.add_argument("--train_xml_dir", dest='train_xml_dir', help=
    "train xmlfile path", default="../Plate_dataset/AC/train/xml/", type=str)
parser.add_argument("--test_xml_dir", dest='test_xml_dir', help=
    "test xmlfile path", default="../Plate_dataset/AC/test/xml/", type=str)

args = parser.parse_args()

for mode in ["train","test"]:
    if mode == "test":
        img_dir = args.test_img_dir
        pos_save_dir = "./data/valPnet/positive"
        part_save_dir = "./data/valPnet/part"
        neg_save_dir = "./data/valPnet/negative"
        anno_save_dir = "./data/valPnet/anno_store"
    else:

        img_dir = args.train_img_dir
        pos_save_dir = "./data/trainPnet/positive"
        part_save_dir = "./data/trainPnet/part"
        neg_save_dir = "./data/trainPnet/negative"
        anno_save_dir = "./data/trainPnet/anno_store"


    print(os.path)
    if not os.path.exists(pos_save_dir):
        try:
            os.mkdir(pos_save_dir)
        except FileNotFoundError:
            print("请创建 " + pos_save_dir)
    if not os.path.exists(anno_save_dir):
        try:
            os.mkdir(anno_save_dir)
        except FileNotFoundError:
            print("请创建 " + anno_save_dir)
    if not os.path.exists(part_save_dir):
        try:
            os.mkdir(part_save_dir)
        except FileNotFoundError:
            print("请创建 " + part_save_dir)
    if not os.path.exists(neg_save_dir):
        try:
            os.mkdir(neg_save_dir)
        except FileNotFoundError:
            print("请创建 " + neg_save_dir)

    # 把不同labels的图片存储在不同的文档，把图片的label，图片地址，label -1和1的图片还有保存offset
    if mode =="test":
        f1 = open('./data/valPnet/anno_store/pos_pnet_val.txt', 'w')
        f2 = open('./data/valPnet/anno_store/neg_pnet_val.txt', 'w')
        f3 = open('./data/valPnet/anno_store/part_pnet_val.txt', 'w')
    else:
        f1 = open('./data/trainPnet/anno_store/pos_pnet_val.txt', 'w')
        f2 = open('./data/trainPnet/anno_store/neg_pnet_val.txt', 'w')
        f3 = open('./data/trainPnet/anno_store/part_pnet_val.txt', 'w')

    img_paths = []
    img_paths += [el for el in paths.list_images(img_dir)]
    random.shuffle(img_paths)
    num = len(img_paths)
    print("%d pics in total" % num)

    p_idx = 0  # positive
    n_idx = 0  # negative
    d_idx = 0  # dont care
    idx = 0
    for annotation in img_paths:
        im_path = annotation
        basename = os.path.basename(im_path)
        num = basename.split(".")[0]
        if mode == "test":
            file_xml = args.test_xml_dir + str(num) + ".xml"
        else:
            file_xml = args.train_xml_dir + str(num) + ".xml"
        anno = ET.ElementTree(file=file_xml)
        xmin = int(anno.find('object').find('bndbox').find('xmin').text)
        ymin = int(anno.find('object').find('bndbox').find('ymin').text)
        xmax = int(anno.find('object').find('bndbox').find('xmax').text)
        ymax = int(anno.find('object').find('bndbox').find('ymax').text)
        # 读取ground truth
        box1 = [xmin, ymin, xmax, ymax]
        x1, y1, x2, y2 = xmin, ymin, xmax, ymax

        boxes = np.zeros((1, 4), dtype=np.int32)
        boxes[0, 0], boxes[0, 1], boxes[0, 2], boxes[0, 3] = x1, y1, x2, y2

        img = cv2.imread(im_path)
        idx += 1

        height, width, channel = img.shape

        neg_num = 0
        while neg_num < 35:
            size_x = np.random.randint(47, min(width, height) / 2)
            size_y = np.random.randint(12, min(width, height) / 2)
            nx = np.random.randint(0, width - size_x)
            ny = np.random.randint(0, height - size_y)
            crop_box = np.array([nx, ny, nx + size_x, ny + size_y])

            Iou = IoU(crop_box, boxes)

            cropped_im = img[ny: ny + size_y, nx: nx + size_x, :]
            resized_im = cv2.resize(cropped_im, (47, 12), interpolation=cv2.INTER_LINEAR)

            if np.max(Iou) < 0.3:
                # 随机获得neg的训练数据
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                f2.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1

        for box in boxes:
            # box (x_left, y_top, w, h)
            x1, y1, x2, y2 = box
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            # generate negative examples that have overlap with gt
            for i in range(5):
                size_x = np.random.randint(47, min(width, height) / 2)
                size_y = np.random.randint(12, min(width, height) / 2)
                # delta_x and delta_y are offsets of (x1, y1)
                delta_x = np.random.randint(max(-size_x, -x1), w)
                delta_y = np.random.randint(max(-size_y, -y1), h)
                nx1 = max(0, x1 + delta_x)
                ny1 = max(0, y1 + delta_y)

                if nx1 + size_x > width or ny1 + size_y > height:
                    continue
                crop_box = np.array([nx1, ny1, nx1 + size_x, ny1 + size_y])
                Iou = IoU(crop_box, boxes)

                cropped_im = img[ny1: ny1 + size_y, nx1: nx1 + size_x, :]
                resized_im = cv2.resize(cropped_im, (47, 12), interpolation=cv2.INTER_LINEAR)

                if np.max(Iou) < 0.3:
                    # 随机获得neg的训练数据
                    save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                    f2.write(save_file + ' 0\n')
                    cv2.imwrite(save_file, resized_im)
                    n_idx += 1
            # 产生pos和part-plate的数据
            for i in range(20):
                size_x = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
                size_y = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

                # delta 是指中心的偏离
                delta_x = np.random.randint(-w * 0.2, w * 0.2)
                delta_y = np.random.randint(-h * 0.2, h * 0.2)

                nx1 = max(x1 + w / 2 + delta_x - size_x / 2, 0)
                ny1 = max(y1 + h / 2 + delta_y - size_y / 2, 0)
                nx2 = nx1 + size_x
                ny2 = ny1 + size_y

                if nx2 > width or ny2 > height:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])

                offset_x1 = (x1 - nx1) / float(size_x)
                offset_y1 = (y1 - ny1) / float(size_y)
                offset_x2 = (x2 - nx2) / float(size_x)
                offset_y2 = (y2 - ny2) / float(size_y)

                cropped_im = img[int(ny1): int(ny2), int(nx1): int(nx2), :]
                resized_im = cv2.resize(cropped_im, (47, 12), interpolation=cv2.INTER_LINEAR)

                box_ = box.reshape(1, -1)
                if IoU(crop_box, box_) >= 0.65:
                    #保存positive的数据
                    save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                    f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1

                elif IoU(crop_box, box_) >= 0.4 and d_idx < 1.2 * p_idx + 1:
                    #保存part的训练数据
                    save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                    f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1

        print("%s数据生成中 %s 张图片处理完成, 产生pos训练集: %s  产生part训练集: %s 产生neg训练集: %s" % (mode,idx, p_idx, d_idx, n_idx))

    f1.close()
    f2.close()
    f3.close()