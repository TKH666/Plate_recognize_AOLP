# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : Tam KaHou
import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from model.cnn_model import create_cnn
import argparse
import time
from imutils import paths

def plate_load(img_dir,xml_dir):
    '''
    加载车牌数据，获取车牌坐标并且截取车牌
    :param img_dir: str 车牌图片路径
    :param xml_dir: str 车牌的xml坐标文件路径
    :return:
        plate:np.array
        shape:[plate_height,plate_width,channel]
    '''
    org_fig = cv.imread(img_dir)
    #读取车牌的坐标
    anno = ET.ElementTree(file=xml_dir)
    xmin = int(anno.find('object').find('bndbox').find('xmin').text)
    ymin = int(anno.find('object').find('bndbox').find('ymin').text)
    xmax = int(anno.find('object').find('bndbox').find('xmax').text)
    ymax = int(anno.find('object').find('bndbox').find('ymax').text)
    label = anno.find('object').find('platetext').text
    print("原车牌为: ",label)
    width = xmax - xmin
    height = ymax - ymin
    plate = np.zeros((height, width))
    plate = org_fig[ymin:ymin + height, xmin:xmin + width]
    return plate

def plate_threshold(plate):
    '''
    对车牌二值化，对车牌进行预处理，去噪，均衡化，再二值化变成黑底白字
    :param plate: np.array 车牌
    :return:
        val_box:np.array
                shape: [[plate_height,plate_width]
    '''
    val_box = plate.copy()
    val_box = cv.cvtColor(val_box, cv.COLOR_BGR2GRAY)
    val_box = cv.equalizeHist(val_box)
    thesh, val_box = cv.threshold(val_box, 90, 255, cv.THRESH_BINARY)
    val_box = np.array(abs(np.array(val_box, dtype=np.int64) - 255), dtype=np.uint8)
    return val_box
    # -----------------------------------------------

def cut_plate(val_box):
    '''
    对车牌的上下和左右边缘的黑色部分进行裁剪，裁剪剩下车牌的主体部分
    :param val_box: np.array
             shape: [[plate_height,plate_width]
    :return:
        val_box_tmp:array
              shape:[plate_height-cutsize,plate_width]
    '''
    hang = [] #用于储存要删除的行
    for i in range(val_box.shape[0]):
        #假如黑色像素的值超过行size的80%会被删除
        if len(val_box[i, :][np.equal(val_box[i, :], 255)]) >= val_box.shape[1] * 0.8:
            hang.append(i)
    val_box_tmp = np.delete(val_box, hang, axis=0)

    # 对车牌的左右两边的列进行扫描，有一点白色直接被判断为噪点，但是不是裁剪，是让这些列变成全黑，为了就是下部按照垂直投影切割车牌字符创建黑色背景
    for i in range(5):
        #左边缘扫描
        # if len(val_box_tmp[:,i][np.equal(val_box_tmp[:,i],255)]) >= val_box_tmp.shape[0]*0.2:
        if len(val_box_tmp[:, i][np.equal(val_box_tmp[:, i], 255)]) != 0:
            val_box_tmp[:, i] = 0

    for i in range(val_box_tmp.shape[1] - 3, val_box.shape[1]):
        #右边缘扫描
        # if len(val_box_tmp[:,i][np.equal(val_box_tmp[:,i],255)]) >= val_box_tmp.shape[0]*0.1:
        if len(val_box_tmp[:, i][np.equal(val_box_tmp[:, i], 255)]) != 0:
            val_box_tmp[:, i] = 0
    return val_box_tmp
    # --------------------------------

def black_white_row_statistic(val_box_tmp):
    '''
    func:以行为单位，统计每一行的黑白像素数目，并且生成水平投影的白色像素个数柱状图，获得白色像素的水平投影分布
    :param val_box_tmp:  shape:[plate_height-cutsize,plate_width]
    :return:
        white_hang:list
        black_hang:list
        mount_hang:np.array,白色像素的水平投影分布
    '''
    white_hang = [0 for i in range(val_box_tmp.shape[0])]
    black_hang = [0 for i in range(val_box_tmp.shape[0])]
    for index in range(val_box_tmp.shape[0]):
        mark_0 = np.equal(val_box_tmp[index, :], 0)
        num_0 = len(val_box_tmp[index, :][mark_0])
        black_hang[index] = num_0
        mark_255 = np.equal(val_box_tmp[index, :], 255)
        num_255 = len(val_box_tmp[index, :][mark_255])
        white_hang[index] = num_255
    mount_hang = np.zeros(val_box_tmp.shape, dtype=np.uint8)
    for i in range(mount_hang.shape[0]):
        mount_hang[i, 0:white_hang[i]] = 255
    white_hang = np.array(white_hang)
    return white_hang,black_hang,mount_hang
    #--------------------------------

def adjus_plate(val_box_tmp,white_hang):
    '''
    进一步调整车牌主体内容的上下边缘，极限切割，尽量保留图片的高，整体的思想是从车牌主体中间往上下扫描，最后得到上下边界
    :param val_box_tmp:
    :param order_index:水平方向白色像素根据
    :return:
    '''
    up_mark = False
    down_mark = False
    mid = int(val_box_tmp.shape[0] / 2)
    order_index = np.argsort(white_hang)
    leng = 0
    xeng = 0
    up_limit = 0
    down_limit = val_box_tmp.shape[0]
    for i in order_index[:12]:
        tmpcm = mid - i
        #从上下进行切割的判断，找准最后的上下界，并且上界和下界的保持图片的16像素高
        if up_mark == True and tmpcm < leng and i <= val_box_tmp.shape[0] / 2 and down_limit - i >= 16:
            up_limit = i
            leng = tmpcm
            # print(up_limit)
        if i <= val_box_tmp.shape[0] / 2 and up_mark == False:
            up_mark = True
            up_limit = i
            leng = mid - i

        dowmcm = i - mid
        if down_mark == True and dowmcm < xeng and i > val_box_tmp.shape[0] / 2 and i - up_limit >= 16:
            down_limit = i
            xeng = dowmcm
        if i > val_box_tmp.shape[0] / 2 and down_mark == False:
            down_mark = True
            down_limit = i
            xeng = i - mid

    return up_limit,down_limit
    # print(up_limit,down_limit)
    #val_box_tmp = val_box_tmp[up_limit:down_limit, :]

    # --------------------------------

def black_white_colmun_statistic(val_box_tmp):
    '''
    把经过进一步的切割的车牌进行垂直的投影，统计垂直方向的白色像素分布并且生成柱状图，用于下一部的车牌分割
    :param val_box_tmp: np.array 车牌
    :return:
        white_lie:list
        black_lie:list
        mount_lie: np.array,白色像素的水平投影分布
    '''
    mount_lie = np.zeros(val_box_tmp.shape, dtype=np.uint8)
    white_lie = [0 for i in range(val_box_tmp.shape[1])]
    black_lie = [0 for i in range(val_box_tmp.shape[1])]
    for index in range(val_box_tmp.shape[1]):
        mark_0 = np.equal(val_box_tmp[:, index], 0)
        num_0 = len(val_box_tmp[:, index][mark_0])
        black_lie[index] = num_0
        mark_255 = np.equal(val_box_tmp[:, index], 255)
        num_255 = len(val_box_tmp[:, index][mark_255])
        white_lie[index] = num_255

    for i in range(mount_lie.shape[1]):
        mount_lie[mount_lie.shape[0] - white_lie[i]:mount_lie.shape[0], i] = 255
    return white_lie,black_lie,mount_lie
    # --------------------------------
    # plt.imshow(cv.cvtColor(val_box_tmp,cv.COLOR_BGR2RGB))
    # plt.show()
    # plt.imshow(cv.cvtColor(mount_hang,cv.COLOR_BGR2RGB))
    # plt.show()
    # plt.imshow(cv.cvtColor(mount_lie,cv.COLOR_BGR2RGB))
    # plt.show()
    # ------------------------------------------------------

def separate_regz_char(white_lie,val_box_tmp):
    '''
    切割车牌的字符并且识别字符
    :param white_lie: list 垂直方向的车牌白色像素的统计分布情况
    :param val_box_tmp: np.array 车牌
    :return:
        plate_text :str 车牌的识别结果
    '''
    #算法的思路是先把低于白色像素阈值的坐标记录放入新的list，然后再通过两两之间的组合，把一个字符的左右边界确定，切割识别
    plate_text = '' #车牌识别内容
    border = [] #储存左右边界的坐标
    threshold = 1 #小于等于一个白色像素作为阈值，被认为是字符的边界，因为是白字黑底，所以黑色的背景要纯粹
    for i in range(len(white_lie)):
        if white_lie[i] <= threshold:
            border.append(i)



    #车牌字符的识别切割，其中因为有时候会有噪点的出现，左右边界的切割并不好，不是一个一个字符切割，会有字符粘连，根据切割图片大小再进行切割


    for i in range(len(border) - 1):
        if border[i + 1] - border[i] > 2 and np.mean(val_box_tmp[:, border[i]:border[i + 1] + 1]) > 50:

            if border[i + 1] + 1 - border[i] > 17 and border[i + 1] + 1 - border[i] <= 30:
                '''
                两个字符粘连的问题，进行切割，根据白色像素的在图中的分布，找到中间的白色像素小于阈值的坐标，作为两个字符粘连图片的二分切割点
                '''
                tmp_list = white_lie[border[i]:border[i + 1] + 1]
                mid = len(tmp_list) // 2 #初始化二等分点
                for k in np.argsort(tmp_list):
                    if abs(mid - k) <= 2:
                        breakpoint = k
                        break
                #plt.imshow(val_box_tmp[:, border[i]:border[i] + breakpoint])
                #切割为单张图片，并且用训练好的网络进行字符识别
                pic = val_box_tmp[:, border[i]:border[i] + breakpoint]
                pic = cv.cvtColor(pic, cv.COLOR_GRAY2BGR)
                char = create_cnn(pic, model_dir="./model/best_model.pt")
                plate_text += char

                # 切割为单张图片，并且用训练好的网络进行字符识别
                pic1 = val_box_tmp[:, border[i + 1] + 1 - breakpoint:border[i + 1] + 1]
                pic1 = cv.cvtColor(pic1, cv.COLOR_GRAY2BGR)
                char = create_cnn(pic1, model_dir="./model/best_model.pt")
                plate_text += char

            elif border[i + 1] + 1 - border[i] > 30 and border[i + 1] + 1 - border[i] < 40:
                '''
                三个字符粘连的问题，进行切割，算法也是根据白色像素的分布，找出三等分点，然后用三等分点分离粘连的图片
                '''
                tmp_list = white_lie[border[i]:border[i + 1] + 1]

                third_one = len(tmp_list) // 3 #初始化三等分点
                t_o, t_t = 0, 0
                t_o_mark, t_t_mark = False, False
                for j in np.argsort(tmp_list):
                    if t_o_mark and t_t_mark:
                        break
                    elif t_o_mark == False and abs(third_one - j) <= 2:
                        t_o = j
                        t_o_mark = True
                    elif t_t_mark == False and abs(third_one * 2 - 1 - j) <= 2:
                        t_t = j
                        t_t_mark = True
                # 切割为单张图片，并且用训练好的网络进行字符识别
                pic1 = val_box_tmp[:, border[i]:border[i] + t_o]
                pic1 = cv.cvtColor(pic1, cv.COLOR_GRAY2BGR)
                char = create_cnn(pic1, model_dir="./model/best_model.pt")
                plate_text += char

                # 切割为单张图片，并且用训练好的网络进行字符识别
                pic2 = val_box_tmp[:, border[i]:border[i] + t_o]
                pic2 = cv.cvtColor(pic2, cv.COLOR_GRAY2BGR)
                char = create_cnn(pic2, model_dir="./model/best_model.pt")
                plate_text += char

                # 切割为单张图片，并且用训练好的网络进行字符识别
                pic3 = val_box_tmp[:, border[i]:border[i] + t_o]
                pic3 = cv.cvtColor(pic3, cv.COLOR_GRAY2BGR)
                char = create_cnn(pic3, model_dir="./model/best_model.pt")
                plate_text += char


            else:
                #单张图片，并且用训练好的网络进行字符识别
                pic = val_box_tmp[:, border[i]:border[i + 1] + 1]
                pic = cv.cvtColor(pic, cv.COLOR_GRAY2BGR)
                char = create_cnn(pic, model_dir="./model/best_model.pt")
                plate_text += char
    return plate_text[:6]

def gen_xml(num,plate_text,xml_pred):
    filename = xml_pred + str(num) + ".xml"
    with open(filename, "w") as fw:
        print("<annotation><object><platetext>{}</platetext></object></annotation>".format(plate_text),file=fw)
        print("第{}张图片写入xml成功".format(num))

def recognize(plate_img):
    '''
    车牌识别
    :param plate_img: np.array  shape:[h,w,c]
    :return: plate_text :str
    '''
    #车牌二值化
    plate_bw = plate_threshold(plate_img)

    #裁剪车牌获得车牌主要字符区域
    plate_bw = cut_plate(plate_bw)

    #黑白像素水平投影车牌的统计
    white_row, black_row, distib_row_img = black_white_row_statistic(plate_bw)

    # 进一步调整车牌的上下区域，减少上下的区域噪音
    up_limit, down_limit = adjus_plate(plate_bw, white_row)
    plate_bw = plate_bw[up_limit:down_limit, :]

    #黑白像素垂直投影车牌的统计
    white_col, black_col, distib_col_img = black_white_colmun_statistic(plate_bw)

    #车牌的字符分割识别
    plate_text = separate_regz_char(white_col, plate_bw)
    return plate_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plate recognize')
    parser.add_argument("--image", dest='image', help=
    "image_file_dir", default="../Plate_dataset/AC/train/jpeg/518.jpg", type=str)
    parser.add_argument("--xml", dest='xml', help=
    "image_xmlfile_dir", default="../Plate_dataset/AC/train/xml/518.xml", type=str)
    parser.add_argument("--gen_xml", dest='gen_xml', help=
    "generate xml or not", default=False, type=bool)
    parser.add_argument("--save_xml_dir", dest='save_xml_dir', help=
    "the path of saving xml ", default="../Plate_dataset/AC/test/xml_pred_recognize", type=str)
    parser.add_argument("--img_dir", dest='img_dir', help=
    "image dir ", default="../Plate_dataset/AC/test/jpeg/", type=str)
    parser.add_argument("--xml_dir", dest='xml_dir', help=
    "image_xmlfile_dir", default="../Plate_dataset/AC/test/xml/", type=str)
    args = parser.parse_args()

    start = time.time()
    img_num= os.path.basename(args.img_dir).split(".")[0]
    img_num2 = os.path.basename(args.xml_dir).split(".")[0]
    if img_num != img_num2:
        raise ValueError("请提供正确的车牌坐标文件，确保xml对应相同的车牌图片")

    file_dir = args.image
    file_xml = args.xml

    plate_img=plate_load(file_dir,file_xml)
    plate_text=recognize(plate_img)

    if args.gen_xml:
        xml_pred_dir = args.save_xml_dir
        img_dir = []
        img_dir+=[el for el in paths.list_images(args.img_dir)]
        xml_dir = args.xml_dir
        for file_path in img_dir:
            num=os.path.basename(args.img_dir).split(".")[0]
            xml_path = args.xml_dir+str(num)+".xml"
            plate_img = plate_load(file_path, xml_path)
            plate_text = recognize(plate_img)
            gen_xml(num,plate_text,xml_pred_dir)
    print("车牌识别为",plate_text)
    print("车牌识别花费了{:2.3f} 秒".format(time.time() - start))
