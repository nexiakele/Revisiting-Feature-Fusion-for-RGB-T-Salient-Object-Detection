
import numpy as np
import cv2
import glob2
import os

datas = os.listdir('saliencymap')#

for data in datas:
    #dirname,filename=os.path.split(train)
    name, ext = os.path.splitext(data)
    newname, stage = name.split('_R')
    image = cv2.imread(os.path.join('saliencymap', data))#big/pedestrian4/grayscale
    save_name = os.path.join('MRCMC', newname + '.png')
    #save_name = os.path.join('Result-2', name + '_' + 'RT_' + '.png')
    cv2.imwrite(save_name, image)
