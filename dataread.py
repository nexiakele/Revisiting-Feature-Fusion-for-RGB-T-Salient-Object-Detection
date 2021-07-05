import numpy as np
import cv2
import glob2
import os

datas = os.listdir('big/pedestrian3/grayscale')#

for data in datas:
    #dirname,filename=os.path.split(train)
    name, ext = os.path.splitext(data)
    image = cv2.imread(os.path.join('big/pedestrian3/grayscale', data))#big/pedestrian4/grayscale
    save_name = os.path.join('DATA/RGB', 'pedestrian3' + data)
    #save_name = os.path.join('Result-2', name + '_' + 'RT_' + '.png')
    cv2.imwrite(save_name, image)

    thermal = cv2.imread(os.path.join('big/pedestrian3/infrared', name + '.png'))
    save_name = os.path.join('DATA/thermal', 'pedestrian3' + name + '.png')
    cv2.imwrite(save_name, thermal)

    gt = cv2.imread(os.path.join('big/pedestrian3/groundtruth', name + '.bmp'))
    save_name = os.path.join('DATA/GT', 'pedestrian3' + name + '.png')
    cv2.imwrite(save_name, gt)
