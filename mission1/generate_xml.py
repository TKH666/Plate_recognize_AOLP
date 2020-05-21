# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : Tam KaHou
import  os
import sys
sys.path.append(os.getcwd())
from MTCNN import create_mtcnn_net
import torch
import  cv2 as cv
import numpy as np
import argparse
from imutils import paths

def generate_xml(file):
    global fail, success
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_dir = file
    num=os.path.basename(file).split(".")[0]
    base_name=os.path.basename(file)
    box=[]
    img=cv.imread(img_dir)
    print("{}加载成功".format(base_name))
    print("MTCNN开始加载图片")
    bboxs=create_mtcnn_net(image=img,mini_lp_size=(50,15),device=device,p_model_path="./model/pnet_Weights",o_model_path="./model/onet_Weights")
    max_prob=0
    if len(bboxs)==0:
        '''
        预测失败，调高亮度
        '''

        b, g, r = cv.split(img)
        b = cv.equalizeHist(b)
        g = cv.equalizeHist(g)
        r = cv.equalizeHist(r)
        img = cv.merge((b, g, r))
        bboxs = create_mtcnn_net(image=img, mini_lp_size=(50, 15), device=device, p_model_path="./model/pnet_Weights",
                                 o_model_path="./model/onet_Weights")
        print("{}第一次预测失败，重新校正预测".format(base_name))

    if len(bboxs)!=1:
        #假如有多个预测框，首选预测概率最大的，然后再检测是否符合车牌面积
        max_prob_index=np.argmax(bboxs[:,4])
        bbox = bboxs[max_prob_index, :4]
        bbox = [int(a) for a in bbox]
        w = int(bbox[2]) - int(bbox[0])
        h = int(bbox[3]) - int(bbox[1])
        if w*h >1300:
            box = bbox
        else:
            #重新根据车牌的大小来筛选边框
            for index in range(len(bboxs)):
                prob=bboxs[index,4]
                bbox= bboxs[index,:4]
                bbox = [int(a) for a in bbox]
                w = int(bbox[2]) - int(bbox[0])
                h = int(bbox[3]) - int(bbox[1])
                if w*h >1300:
                    if max_prob<prob:
                        box=bbox
                        max_prob=prob
    else:
        box=bboxs[0,:4]
        box=[int(a) for a in box]
    if len(box) ==0:
        fail +=1
        box=[0,0,0,0]
    xmin,ymin,xmax,ymax=box[0],box[1],box[2],box[3]
    print("{} 模型预测成功".format(base_name))
    success+=1
    xml_pred = args.xml_save_dir
    filename = xml_pred+"/"+str(num)+".xml"
    with open(filename,"w") as fw:
        print("<annotation><object><bndbox>",file=fw)
        print("<xmin>{}</xmin><ymin>{}</ymin><xmax>{}</xmax><ymax>{}</ymax>".format(xmin,ymin,xmax,ymax),file=fw)
        print("</bndbox></object></annotation>",file=fw)
        print("第{}张图片写入xml成功".format(num))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate xml_file')
    parser.add_argument("--image_dir", dest='img_dir', help=
    "image path", default="./Plate_dataset/AC/test/jpeg/", type=str)
    parser.add_argument("--xml_save_dir", dest='xml_save_dir', help=
    "save xml_file path", default="../Plate_dataset/AC/test/xml_pred_detection/", type=str)
    args = parser.parse_args()
    img_list=[]
    img_list+=[fl for fl in paths.list_images(args.img_dir)]
    fail, success=0,0
    for file in img_list:
        generate_xml(file)
    print("{}个xml文件成功生成，生成失败xml {}个,总生成{}".format(success,fail,len(img_list)))