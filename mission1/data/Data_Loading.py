# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : Tam KaHou
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv

class ListDataset(Dataset):
    '''
    用于加载训练数据,pytorch的Dataset方法重写
    '''
    def __init__(self, list_path):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):

        annotation = self.img_files[index % len(self.img_files)].strip().split(' ')


        #  Image加载

        img = cv.imread(annotation[0])
        img = img[:,:,::-1]
        img = np.asarray(img, 'float32')
        img = img.transpose((2, 0, 1))
        img = (img - 127.5) * 0.0078125
        input_img = torch.FloatTensor(img)

        #  Label加载

        label = int(annotation[1])
        bbox_target = np.zeros((4,))
        landmark = np.zeros((10,))

        if len(annotation[2:]) == 4:
            #label = 0 （iou<0.3）的图片
            bbox_target = np.array(annotation[2:6]).astype(float)
        if len(annotation[2:]) == 14:
            # label = 1/-1 （iou>0.6）( 0.4<iou<0.6)的图片
            bbox_target = np.array(annotation[2:6]).astype(float)
            landmark = np.array(annotation[6:]).astype(float)

        sample = {'input_img': input_img, 'label': label, 'bbox_target': bbox_target, 'landmark': landmark}

        return sample

if __name__ == '__main__':

    train_path = './data/trainPnet/anno_store/imglist_anno_12.txt'
    val_path = './data/valPnet/anno_store/imglist_anno_12_val.txt'
    batch_size = 8
    dataloaders = {'train': torch.utils.data.DataLoader(ListDataset(train_path), batch_size=batch_size, shuffle=True),
                   'val': torch.utils.data.DataLoader(ListDataset(val_path), batch_size=batch_size, shuffle=True)}
    dataset_sizes = {'train': len(ListDataset(train_path)), 'val': len(ListDataset(val_path))}

    for i_batch, sample_batched in enumerate(dataloaders['train']):

        images_batch, label_batch, bbox_batch, landmark_batch = sample_batched['input_img'], sample_batched[
            'label'], sample_batched['bbox_target'], sample_batched['landmark']

        print(i_batch, images_batch.shape, label_batch.shape, bbox_batch.shape, landmark_batch.shape)

        if i_batch == 3:
            break