# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : Tam KaHou
import sys
import cv2 as cv
import os
sys.path.append(os.getcwd())
import numpy as np
import xml.etree.ElementTree as ET
from utilize import *
import torch
import random
from imutils import paths
from MTCNN import create_mtcnn_net
import argparse

#生成Onet的训练集和预测集们，根据不同的任务产生不同IOU的数据

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

for mode in ['train','test']:
    if mode =='test':
        img_dir = args.test_img_dir
        pos_save_dir = "./data/valOnet/positive"
        part_save_dir = "./data/valOnet/part"
        neg_save_dir = "./data/valOnet/negative"
        anno_save_dir= "./data/valOnet/anno_store"
    else:
        img_dir = args.train_img_dir
        pos_save_dir = "./data/trainOnet/positive"
        part_save_dir = "./data/trainOnet/part"
        neg_save_dir = "./data/trainOnet/negative"
        anno_save_dir = "./data/trainOnet/anno_store"

    if not os.path.exists(pos_save_dir):
        try:
            os.mkdir(pos_save_dir)
        except FileNotFoundError:
            print("请创建 "+pos_save_dir)
    if not os.path.exists(anno_save_dir):
        try:
            os.mkdir(anno_save_dir)
        except FileNotFoundError:
            print("请创建 "+anno_save_dir)
    if not os.path.exists(part_save_dir):
        try:
            os.mkdir(part_save_dir)
        except FileNotFoundError:
            print("请创建 "+part_save_dir)
    if not os.path.exists(neg_save_dir):
        try:
            os.mkdir(neg_save_dir)
        except FileNotFoundError:
            print("请创建 " + neg_save_dir)

    # 把不同labels的图片存储在不同的文档，把图片的label，图片地址，label -1和1的图片还有保存offset
    if mode =="test":
        pos_doc = open('./data/valOnet/anno_store/pos_onet_val.txt', 'w')
        neg_doc = open('./data/valOnet/anno_store/neg_onet_val.txt', 'w')
        part_doc = open('./data/valOnet/anno_store/part_onet_val.txt', 'w')
    else:
        pos_doc = open('./data/trainOnet/anno_store/pos_onet_val.txt', 'w')
        neg_doc = open('./data/trainOnet/anno_store/neg_onet_val.txt', 'w')
        part_doc = open('./data/trainOnet/anno_store/part_onet_val.txt', 'w')

    img_paths = []
    img_paths += [el for el in paths.list_images(img_dir)]
    random.shuffle(img_paths)
    num = len(img_paths)
    print("%d pics in total" % num)

    image_size = (94, 24)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    p_idx = 0  # positive iou >0.65
    n_idx = 0  # negative iou<0.3
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
        #读取ground truth
        box1 = [xmin, ymin, xmax, ymax]
        x1, y1, x2, y2 = xmin, ymin, xmax, ymax


        boxes = np.zeros((1, 4), dtype=np.int32)
        boxes[0, 0], boxes[0, 1], boxes[0, 2], boxes[0, 3] = x1, y1, x2, y2

        image = cv.imread(im_path)
        #经过pnet训练得到候选框
        bboxes = create_mtcnn_net(image, (50,15), device, p_model_path='./model/pnet_Weights', o_model_path=None)
        dets = np.round(bboxes[:, 0:4])

        if dets.shape[0] == 0:
            continue

        img = cv.imread(im_path)
        idx += 1

        height, width, channel = img.shape

        for box in dets:
            x_left, y_top, x_right, y_bottom = box[0:4].astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # 忽略那些框太小或者超出图片的box
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue

            # 计算pnet的候选框和真正车牌的box的iou
            Iou = IoU(box, boxes)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv.resize(cropped_im, image_size, interpolation=cv.INTER_LINEAR)

            # 保存negtivate图片的 label和图片地址
            if np.max(Iou) < 0.3 and n_idx < 3.2 * p_idx + 1:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                neg_doc.write(save_file + ' 0\n')
                cv.imwrite(save_file, resized_im)
                n_idx += 1
            else:
                # 找到ground truth的最高iou的box
                idx_Iou = np.argmax(Iou)
                assigned_gt = boxes[idx_Iou]
                x1, y1, x2, y2 = assigned_gt

                # 计算 bbox reg label
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                # 保存positive和part-face的图片和label
                if np.max(Iou) >= 0.65:
                    save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                    pos_doc.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv.imwrite(save_file, resized_im)
                    p_idx += 1

                elif np.max(Iou) >= 0.4 and d_idx < 1.2 * p_idx + 1:
                    save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                    part_doc.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv.imwrite(save_file, resized_im)
                    d_idx += 1

        print("%s数据生成中， %s 张图片处理完成, 产生pos训练集: %s  产生part训练集: %s 产生neg训练集: %s" % (mode,idx, p_idx, d_idx, n_idx))

    pos_doc.close()
    neg_doc.close()
    part_doc.close()