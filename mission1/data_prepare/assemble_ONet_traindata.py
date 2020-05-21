# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : Tam KaHou
import numpy.random as npr
import numpy as np

import os
import sys
sys.path.append(os.getcwd())
def assemble_data(output_file, anno_file_list=[]):

    #把生成的pos，neg，part的训练数据组装一起

    if len(anno_file_list)==0:
        return 0

    if os.path.exists(output_file):
        os.remove(output_file)

    for anno_file in anno_file_list:
        with open(anno_file, 'r') as f:
            anno_lines = f.readlines()
        idx_keep = np.arange(len(anno_lines))
        np.random.shuffle(idx_keep)
        chose_count = 0
        with open(output_file, 'a+') as f:
            for idx in idx_keep:
                # 把 pos, neg, part 的图片做一个标记写入文件
                f.write(anno_lines[idx])
                chose_count+=1

    return chose_count

if __name__ == '__main__':
    for mode in ["train","test"]:
        anno_list = []
        if mode =="train":
            onet_postive_file = './data/trainOnet/anno_store/pos_onet_val.txt'
            onet_part_file = './data/trainOnet/anno_store/part_onet_val.txt'
            onet_neg_file = './data/trainOnet/anno_store/neg_onet_val.txt'
            imglist_filename = './data/trainOnet/anno_store/imglist_anno_onet.txt'
        else:
            onet_postive_file = './data/valOnet/anno_store/pos_onet_val.txt'
            onet_part_file = './data/valOnet/anno_store/part_onet_val.txt'
            onet_neg_file = './data/valOnet/anno_store/neg_onet_val.txt'
            imglist_filename = './data/valOnet/anno_store/imglist_anno_onet_val.txt'


        anno_list.append(onet_postive_file)
        anno_list.append(onet_part_file)
        anno_list.append(onet_neg_file)

        chose_count = assemble_data(imglist_filename ,anno_list)
        print("ONet train annotation result file path:%s" % imglist_filename)
