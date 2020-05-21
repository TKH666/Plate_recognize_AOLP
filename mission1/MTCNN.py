# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : Tam KaHou
import sys
import os
sys.path.append(os.getcwd())
import argparse
import torch
from model.model_net import PNet, ONet
import math
import numpy as np
from utilize import *
import cv2 as cv
import matplotlib.pyplot as plt
import time

def generate_box(probs,offsets,thresholds,scale):
    '''
    根据网络输出的特征图进行反采样从而得到人脸边框
    :param probs:
    :param offsets:
    :param thresholds:
    :param scale:
    :return:
        bboxes: shape [n, 6]
            前四组值是候选框的坐标[x1,y1,x2,y2]
            第五组是特征图对应的像素的概率值
            第六组是人脸边框的回归值，用于修正预测出来的人脸边框
    '''
    #因为pnet只有一个池化层而且尺寸是(2,5),所以x坐标对应的strid是2，y坐标对应的stride是5，再根据公式
    #W1=Stride*(W2)−2*P+F F对应的是卷积核的size，其实pnet就相对于12x12的窗口一直在扫描图片卷积
    #左上角横坐标 x1=Stride*(W2)
    #右下角 x2=Stride*(W2)
    #左下角 y1=Stride*(W2)+cellsize
    #右下角 y2=Stride*(W2)+cellsize
    stride = (2, 5)
    cell_size = (12, 44)
    # 筛选出大于设定阈值概率的预测结果的坐标（x，y），只有两维，对应x轴下标和y轴下标，对应特征图的坐标
    inds = np.where(probs > thresholds)

    if inds[0].size == 0:
        boxes = None
    else:
        #offsets是pnet人脸边框的回归值，用于修正预测出来的人脸边框
        tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]
        offsets = np.array([tx1, ty1, tx2, ty2])
        #特征图中每一个像素通过模型预测为车牌的概率
        score = probs[inds[0], inds[1]]
        '''
        P-Net用于缩放图片，对应的候选框矩形要进行对应的缩放还原，所以除以scale
        bounding_box前四组值是候选框的坐标[x1,y1,x2,y2]
        第五组是特征图对应的像素的概率值
        第六组是人脸边框的回归值，用于修正预测出来的人脸边框
        '''
        bounding_box = np.vstack([
            np.round((stride[1] * inds[1] + 1.0) / scale),
            np.round((stride[0] * inds[0] + 1.0) / scale),
            np.round((stride[1] * inds[1] + 1.0 + cell_size[1]) / scale),
            np.round((stride[0] * inds[0] + 1.0 + cell_size[0]) / scale),
            score, offsets])
        boxes = bounding_box.T

        #nms算法，去除重叠的候选框的特征图像素
        keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
        boxes[keep]
    return boxes

def create_mtcnn_net(image, mini_lp_size, device, p_model_path=None, o_model_path=None):
    '''
    结合pnet和onet预测车牌的边框坐标
    :param image: np.arrray
    :param mini_lp_size: 最小的[width，height]，
    :param device: pytorch.device
    :param p_model_path: 训练好pnet的模型参数保存目录
    :param o_model_path: 训练好Onet的模型参数保存目录
    :return: bboxes
            np.array
            [车牌的坐标x1,车牌的坐标y1,车牌的坐标x2,车牌的坐标y2,概率]
    '''
    bboxes = np.array([])

    if p_model_path is not None:
        pnet = PNet().to(device)
        #加载pnet训练好的参数
        pnet.load_state_dict(torch.load(p_model_path, map_location=lambda storage, loc: storage))
        #预测模式
        pnet.eval()
        #pnet产生的候选框坐标以及对应的车牌概率
        bboxes = detect_pnet(pnet, image, mini_lp_size, device)
        #print(bboxes)

    if o_model_path is not None:
        onet = ONet().to(device)
        # 加载Onet训练好的参数
        onet.load_state_dict(torch.load(o_model_path, map_location=lambda storage, loc: storage))
        # 预测模式
        onet.eval()
        #Onet最后输出
        bboxes = detect_onet(onet, image, bboxes, device)

    return bboxes

def detect_pnet(pnet, image, min_lp_size, device):
    '''
    用于pnet快速产生初选车牌的候选框
    :param pnet:Pnet pytorch初始化模型
    :param image: np.arrray
    :param mini_lp_size: 最小的[width，height]，
    :param device: pytorch.device
    :return: bboxex,候选框的坐标和对应的概率
    '''
    # start = time.time()

    thresholds = 0.6 # pnet初选的车牌概率 thresholds
    nms_thresholds = 0.7

    # 构造图像金字塔
    height, width, channel = image.shape
    min_height, min_width = height, width

    # 图片缩放的缩放因子
    factor = 0.707


    scales = []

    #创建图片金字塔，首先计算不断resize图片达到最小的宽度和高度
    factor_count = 0
    while min_height > min_lp_size[1] and min_width > min_lp_size[0]:
        scales.append(factor ** factor_count) #f^2，f^3,f^4,f^5...，幂上升缩放
        min_height *= factor
        min_width *=factor
        factor_count += 1

    # it will be returned
    bounding_boxes = []
    #print(factor_count)
    with torch.no_grad():
        # 不同缩放大小下运行P-Net
        for scale in scales:
            sw, sh = math.ceil(width * scale), math.ceil(height * scale)
            img = cv.resize(image, (sw, sh), interpolation=cv.INTER_LINEAR)
            img = torch.FloatTensor(img_preprocess(img)).to(device) #图片处理，把图片转成tensor，并且把CHW改成HWC
            #Pnet预测，返回一张图片每一个像素点的车牌与真正车牌的offset（偏移量）以及是车牌的概率
            offset, prob = pnet(img)
            # probs: 每一个窗口是车牌的概率
            probs = prob.cpu().data.numpy()[0, 1, :, :]
            # offsets: 和真正车牌的bounding boxes的偏移量
            offsets = offset.cpu().data.numpy()

            #Pnet输出的特征图进行反采样从而得到人脸边框，根据特征图像素对应的概率产生候选框坐标切去除重叠候选框
            boxes=generate_box(probs,offsets,thresholds,scale)

            bounding_boxes.append(boxes)

        # 整理不同缩放尺度下产生的bbox
        bounding_boxes = [i for i in bounding_boxes if i is not None]

        #对不同缩放尺寸的候选框进一步去重叠
        if bounding_boxes != []:

            bounding_boxes = np.vstack(bounding_boxes)
            keep = nms(bounding_boxes[:, 0:5], nms_thresholds)
            bounding_boxes = bounding_boxes[keep]
        else:
            bounding_boxes = np.zeros((1,9))

        # 进一步修正候选框的坐标，使与原图像的坐标更接近
        bboxes = adjust_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
        # shape [n_boxes, 5],  x1, y1, x2, y2, score

        bboxes[:, 0:4] = np.round(bboxes[:, 0:4])

        # print("pnet predicted in {:2.3f} seconds".format(time.time() - start))
        global image1

        image1=image.copy()
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, :4]
            cv.rectangle(image1, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        #plt.imshow(image1)
        #plt.show()

        return bboxes

def detect_onet(onet, image, bboxes, device):
    '''
    用于进一步筛选pnet快速产生初选车牌的候选框
    :param onet: Onet pytorch初始化的模型
    :param image: np.arrray
    :param bboxes: 候选框的坐标和对应的概率
    :param device: pytorch.device
    :return:
        bboxes: 候选框的坐标和对应的概率
    '''
    # start = time.time()

    size = (94,24)
    thresholds = 0.8  # 车牌概率thresholds
    nms_thresholds = 0.7
    height, width, channel = image.shape

    num_boxes = len(bboxes)

    # pnet产生的候选框要在图片中取出
    #过滤掉一些超出图片的候选框，返回最后要喂入onet的候选框
    #dx,dy, edx,edy: 抠出来的图的坐标xy，左右坐标的水平差，上下坐标的垂直差
    #y, ey, x, ex :pnet的产生的左下角坐标xy，左右坐标之水平差，上下坐标垂直方向差
    # w, h 宽度，高度
    [dy, edy, dx, edx, y, ey, x, ex, w, h] = fix_bboxes(bboxes, width, height)

    img_boxes = np.zeros((num_boxes, 3, size[1], size[0]))

    for i in range(num_boxes):
        img_box = np.zeros((h[i], w[i], 3))
        #依次从图片中截取候选框
        img_box[dy[i]:(edy[i] + 1), dx[i]:(edx[i] + 1), :] = image[y[i]:(ey[i] + 1), x[i]:(ex[i] + 1), :]

        # resize
        img_box = cv.resize(img_box, size, interpolation=cv.INTER_LINEAR)

        img_boxes[i, :, :, :] = img_preprocess(img_box)

    img_boxes = torch.FloatTensor(img_boxes).to(device)
    offset, prob = onet(img_boxes)
    offsets = offset.cpu().data.numpy()  # shape [n_boxes, 4]
    probs = prob.cpu().data.numpy()  # shape [n_boxes, 2]

    #subset：满足车牌概率的候选框被取出
    plate_box = np.where(probs[:, 1] > thresholds)[0]
    bboxes = bboxes[plate_box]

    #满足车牌概率的bbox的概率以及预测的候选框回归值被取出
    bboxes[:, 4] = probs[plate_box, 1].reshape((-1,))  # assign score from stage 2
    offsets = offsets[plate_box]

    #进一步修正候选框的坐标以及去重叠的候选框，使与原图像的车牌坐标更接近
    bboxes = adjust_box(bboxes, offsets) #回归得到的车牌offset加上 bounding box回归算法得到预测的车牌边框坐标
    keep_box_idx = nms(bboxes, nms_thresholds, mode='min') #对得到的边框去重叠
    bboxes = bboxes[keep_box_idx]
    bboxes[:, 0:4] = np.round(bboxes[:, 0:4])
    # print("onet predicted in {:2.3f} seconds".format(time.time() - start))

    return bboxes

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser=argparse.ArgumentParser(description='MTCNN demo')
    parser.add_argument("--image", dest='img_file', help=
    "image_file_dir", default="../Plate_dataset/AC/train/jpeg/22.jpg", type=str)
    args=parser.parse_args()
    image = cv.imread(args.img_file)
    img=image.copy()
    start = time.time()
    bboxes = create_mtcnn_net(image, (50, 15), device, p_model_path='./model/pnet_Weights', o_model_path="./model/onet_Weights")
    print("车牌检测花了 {:2.3f} 秒".format(time.time() - start))
    max_prob=0
    if len(bboxes)!=1:
        #假如有多个预测框，首选预测概率最大的，然后再检测是否符合车牌面积
        max_prob_index=np.argmax(bboxes[:,4])
        bbox = bboxes[max_prob_index, :4]
        bbox = [int(a) for a in bbox]
        w = int(bbox[2]) - int(bbox[0])
        h = int(bbox[3]) - int(bbox[1])
        if w*h >1300:
            box = bbox
        else:
            #不符合车牌面积的，重新根据车牌的大小来筛选边框
            for index in range(len(bboxes)):
                prob=bboxes[index,4]
                bbox= bboxes[index,:4]
                bbox = [int(a) for a in bbox]
                w = int(bbox[2]) - int(bbox[0])
                h = int(bbox[3]) - int(bbox[1])
                if w*h >1300:
                    if max_prob<prob:
                        box=bbox
                        max_prob=prob
    else:
        box=bboxes[0,:4]
        box=[int(a) for a in box]

    cv.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)

    plt.subplot(1,3,1)
    plt.axis("off")
    plt.title("Original")
    plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))
    plt.subplot(1, 3, 2)
    plt.title("MTCNN_Pnet")
    plt.axis("off")
    plt.imshow(cv.cvtColor(image1,cv.COLOR_BGR2RGB))
    plt.subplot(1, 3, 3)
    plt.title("MTCNN_Onet")
    plt.axis("off")
    plt.imshow(cv.cvtColor(image,cv.COLOR_BGR2RGB))
    plt.show()
