# 因此，每幅图像都产生了真值。用高斯内核把给定图像中的每个人的头部变模糊。
# 所有图像都被裁剪成9个小块，每块的大小是原始图像大小的1/4。
# 将前4个小块均匀裁剪，其余5个小块随机裁剪。最后，每个小块的镜像用于加倍训练集。

import random
import os
from PIL import Image,ImageFilter,ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2

def load_data(img_path,train = True):
    gt_path = img_path.replace('.jpg','.h5').replace('images','ground_truth')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    if train:
        crop_size = (img.size[0]//2, img.size[1]//2)
        #if random.randint(0,9) <= 4:
        if random.randint(0,9) <= 4.5:
        #if random.randint(0,9) <= -1:
            
            dx = int(random.randint(0,1)*img.size[0]*1./2)
            dy = int(random.randint(0,1)*img.size[1]*1./2)
        else:
            dx = int(random.random()*img.size[0]*1./2)
            dy = int(random.random()*img.size[1]*1./2)

        img = img.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
        target = target[dy:crop_size[1]+dy,dx:crop_size[0]+dx]
        
        #if random.random()> 0.8:
        if random.random() > 0.5:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

#由于CSRNet中，网络回归的密度图为原图的1/8，因此作者对密度图进行了降采样，
# 并点乘64以保证密度图之和依然约等于总人数。

    target = cv2.resize(target,(target.shape[1]//8,target.shape[0]//8),interpolation = cv2.INTER_CUBIC)*64
    #target = cv2.resize(target,(target.shape[1],target.shape[0]),interpolation = cv2.INTER_CUBIC)
    return img,target