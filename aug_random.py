import numpy as np
import cv2
import random
import os

def random_crop(x,y): #,z
    h,w = y.shape#[:-1]
    randh = np.random.randint(h/8)
    randw = np.random.randint(w/8)
    randf = np.random.randint(10)
    offseth = 0 if randh == 0 else np.random.randint(randh)
    offsetw = 0 if randw == 0 else np.random.randint(randw)
    p0, p1, p2, p3 = offseth,h+offseth-randh, offsetw, w+offsetw-randw
    if randf >= 5:
        x = x[::, ::-1, ::]
        y = y[::, ::-1]#y[::, ::-1]
        #z = z[::, ::-1, ::]
    return x[p0:p1,p2:p3],y[p0:p1,p2:p3]#,z[p0:p1,p2:p3]

def random_light(x): #,y
    contrast = np.random.rand(1)+0.5
    light = np.random.randint(-20,20)
    x = contrast*x + light
    return np.clip(x,0,255)



images = os.listdir('datas/thermal')
for image in images:
    #if image.find('light') > -1:
        #continue
    path = os.path.join('datas/thermal', image)
    img = cv2.imread(path).astype(np.float32)
    name, ext = os.path.splitext(image)
   # path_t = os.path.join('dataRT/thermal', name + '.png')
    #img_t = cv2.imread(path_t).astype(np.float32)
    path_m = os.path.join('datas/gthermal', name + '.png')
    img_m = cv2.imread(path_m).astype(np.float32)

    #x,y = random_crop(img, img_m)
    x= random_light(img)

    #x = cv2.resize(x, (384, 384))
    #y = cv2.resize(y, (384, 384))
    #Z = cv2.resize(img_m, (384, 384))

    save_name = os.path.join('datas/thermal', 'light_' + image)
    cv2.imwrite(save_name, x.astype(np.uint8))

    #save_name_t = os.path.join('dataRT/thermal', 'light_'+name+'.png')
    #cv2.imwrite(save_name_t, y.astype(np.uint8))

    save_name_m = os.path.join('datas/gthermal', 'light_' + name + '.png')
    cv2.imwrite(save_name_m, img_m.astype(np.uint8))


