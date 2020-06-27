#!/usr/bin/python
#coding:utf-8
import random
import glob
import os
import numpy as np
import cv2
import io
import torch
import random

class FaceDataSet():
    def __init__(self, image_set, is_train=True, batch_size=10):
        self.image_set = image_set
        self.is_train = is_train
        self.batch_size = batch_size
        self.num_samples = 0
        self.dataset = self._get_db()
        self._shuffle_db()
        self.index = 0

    def _get_db(self):
        gt_db = self._load_annotations()
        return gt_db

    def _shuffle_db(self):#shuffle:随机
        idx = random.sample(range(self.num_samples), self.num_samples)
        self.dataset = [self.dataset[0][idx], self.dataset[1][idx]]

    def _load_annotations(self):#anotation:加载数据
        class_names = os.listdir(self.image_set)
        data_x = []
        data_y = []
        label_count = 0
        for name in class_names:
            if self.is_train:
                img_dirs = glob.glob(os.path.join(
                    self.image_set, name)+'/*.pgm')
            else:
                img_dirs = glob.glob(os.path.join(self.image_set, name) + '/*.pgm') + \
                    glob.glob(os.path.join(self.image_set, name)+'/*.jpg')
            
            for i in range(8):
                for item in img_dirs:
                    im = cv2.imread(item)
                    if self.is_train:
                        im=self.rotate_proceed(im,45*i)
                    im = im.astype(np.float32)
                    im = np.transpose(im, (2,0,1))
                    data_x.append(im)
                data_y+= [label_count]*len(img_dirs)
            
            if self.is_train:
                for i in range(8):
                    for item in img_dirs:
                        im = cv2.imread(item)
                        im = cv2.flip(im,1,dst=None) 
                        im=self.rotate_proceed(im,45*i)
                        im = im.astype(np.float32)
                        im = np.transpose(im, (2,0,1))
                        data_x.append(im)
                    data_y+= [label_count]*len(img_dirs)
            label_count += 1
        self.num_samples = len(data_x)
        
        return [np.asarray(data_x), np.asarray(data_y)]

    def get_batch_data(self):
        index = self.index
        index = 0 if index < 0 else index
        index = 0 if index >= self.num_samples else index
        batch_data_x = []
        batch_data_y = []
        end_index = index+self.batch_size if index + \
            self.batch_size < self.num_samples else self.num_samples
        batch_data_x.append(self.dataset[0][index:end_index])
        batch_data_y.append(self.dataset[1][index:end_index])
        if end_index - index < self.batch_size:
            end_index = self.batch_size - end_index + index
            batch_data_x.append(self.dataset[0][0:end_index])
            batch_data_y.append(self.dataset[1][0:end_index])
        if end_index >= self.num_samples:
            end_index = 0
        self.index = end_index
        return np.vstack(batch_data_x).astype(np.float32), np.vstack(batch_data_y).astype(np.int64).reshape((-1))
        #vstack:沿着竖直方向将矩阵堆叠起来。
    def rotate_proceed(self,img,angle):
        imgInfo = img.shape
        height= imgInfo[0]
        width = imgInfo[1]
        deep = imgInfo[2]
        M=cv2.getRotationMatrix2D((width*0.5, height*0.5), angle, 1)
        return cv2.warpAffine(img, M, (width, height))

im_train_set = '/home/crisps/datasets/att_faces_train'
im_val_set = '/home/crisps/datasets/att_faces_val'
is_train = True
batch_size = 80
if __name__ == '__main__':
    db = FaceDataSet(im_train_set, is_train, batch_size)
    
