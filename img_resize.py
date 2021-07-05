import tensorflow as tf
import numpy as np
import glob
import os
import cv2

images = os.listdir('/home/ly/disk2/xiao/RGBT2/data/GT-test546')
for image in images:

    path = os.path.join('/home/ly/disk2/xiao/RGBT2/data/GT-test546', image)
    img = cv2.imread(path)
    img_shape = img.shape
    path1 = os.path.join('MRCMC', image)
    img1 = cv2.imread(path1)
    img2 = cv2.resize(img1, (img_shape[1], img_shape[0]))
    save_name = os.path.join('MR', image)
        #cv2.imwrite('./Datasets/train/aug_testlabels/aug_%06d.bmp'%(i),image)
        #i+=1
    cv2.imwrite(save_name,img2)

