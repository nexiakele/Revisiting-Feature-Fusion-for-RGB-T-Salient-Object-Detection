import numpy as np
import cv2
import glob2
import os

#data = glob2.iglob('data/RGB')
data = os.listdir('data/RGB')
 
length = len(data)
#print(length)
index = 625
trains = []
tests = []
    
indexes = np.array(range(0,length))
np.random.shuffle(indexes)
trn_idxes = indexes[0:index]

tst_idxes = indexes[index:length]

for i in trn_idxes:   
    trains.append(data[i])
for j in tst_idxes:
    tests.append(data[j])

for train in trains:
    #dirname,filename=os.path.split(train)
    name, ext = os.path.splitext(train)
    image = cv2.imread(os.path.join('data/RGB', train))
    save_name = os.path.join('RGB', train)
    cv2.imwrite(save_name, image)

    thermal = cv2.imread(os.path.join('data/thermal', name + '.png'))
    save_name = os.path.join('thermal', name + '.png')
    cv2.imwrite(save_name, thermal)

    gt = cv2.imread(os.path.join('data/GT', name + '.png'))
    save_name = os.path.join('GT', name + '.png')
    cv2.imwrite(save_name, gt)

for test in tests:
    #dirname,filename=os.path.split(test)
    name, ext = os.path.splitext(test)
    image = cv2.imread(os.path.join('data/RGB', test))
    save_name = os.path.join('RGB-test', test)
    cv2.imwrite(save_name, image)

    thermal = cv2.imread(os.path.join('data/thermal', name + '.png'))
    save_name = os.path.join('thermal-test', name + '.png')
    cv2.imwrite(save_name, thermal)

    gt = cv2.imread(os.path.join('data/GT', name + '.png'))
    save_name = os.path.join('GT-test', name + '.png')
    cv2.imwrite(save_name, gt)


