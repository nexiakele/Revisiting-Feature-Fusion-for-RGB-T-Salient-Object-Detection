import tensorflow as tf
import numpy as np
import glob
import os
import cv2

#os.mkdir('RGB_90')
#os.mkdir('RGB_180')
#os.mkdir('RGB_270')

#os.mkdir('RGB_90')
#os.mkdir('RGB_90')
#os.mkdir('RGB_90')
#os.mkdir('RGB_90')

paths=glob.glob('DATA/GT/*')
    # paths=glob.glob('./Datasets/train/m/VIS/*')
print('increase data.........')
#print(paths)
    #i=0
for path in paths:
        
        #sprint(path)
        image = cv2.imread(path)
        dirname,filename=os.path.split(path)
        img_1_90 = np.rot90(image)
        img_1_180 = np.rot90(img_1_90)
        img_1_270 = np.rot90(img_1_180)

        #######flip image
        img_flip1 = cv2.flip(image,1)   #横向翻转图像
        img_1_90_h = np.rot90(img_flip1)
        img_1_180_h = np.rot90(img_1_90_h)
        img_1_270_h = np.rot90(img_1_180_h)
        
        #i+=1
        save_name = os.path.join('DATA/GT', '90_' + filename)
        #cv2.imwrite('./Datasets/train/aug_testimages/aug_%06d.bmp'%(i),image)
        #i+=1
        cv2.imwrite(save_name,img_1_90)
        #i+=1
        save_name = os.path.join('DATA/GT', '180_' + filename)
        cv2.imwrite(save_name,img_1_180)
        #i+=1
        save_name = os.path.join('DATA/GT', '270_' + filename)
        cv2.imwrite(save_name,img_1_270)
        
        
        #i+=1
        save_name = os.path.join('DATA/GT', 'flip_' + filename)
        cv2.imwrite(save_name,img_flip1)
        #i+=1
        save_name = os.path.join('DATA/GT', 'flip_90_' + filename)
        cv2.imwrite(save_name,img_1_90_h)
        #i+=1
        save_name = os.path.join('DATA/GT', 'flip_180_' + filename)
        cv2.imwrite(save_name,img_1_180_h)
        #i+=1
        save_name = os.path.join('DATA/GT', 'flip_270_' + filename)
        cv2.imwrite(save_name,img_1_270_h)
    
print('completed.......')
    