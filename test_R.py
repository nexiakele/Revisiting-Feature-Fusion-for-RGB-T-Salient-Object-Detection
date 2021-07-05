import cv2
import numpy as np
import T_train
import os
import sys
import tensorflow as tf
import time
import vgg16
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#img_t_mean=[101.515, 78.315, 140.606]
#img_t_mean=[85.971, 56.608, 151.944]
#img_t_mean=[127.493, 126.314, 127.453] #small

def load_img_list(dataset):

    if dataset == 'MSRA-B':
        path = 'dataset/MSRA-B/image'
    elif dataset == 'DUT-OMRON':
        path = 'dataset/DUT-OMRON/DUT-OMRON-image'
    
    imgs = os.listdir(path)

    return path, imgs

def image_entropy(input):
        tmp = []  
        for i in range(256):  
            tmp.append(0)  
        val = 0  
        k = 0  
        res = 0  
        #image = input.convert('L')
        img = np.array(input)  
        for i in range(len(img)):  
           for j in range(len(img[i])):  
              val = img[i][j]  
              tmp[val] = float(tmp[val] + 1)  
              k =  float(k + 1)  
        for i in range(len(tmp)):  
            tmp[i] = float(tmp[i] / k)  
        for i in range(len(tmp)):  
           if(tmp[i] == 0):  
             res = res  
           else:  
             res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0))) 
        res_ = res / 8.0 
        return res_



if __name__ == "__main__":

    model = T_train.Model()
    model.build_model()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    img_size = T_train.img_size
    label_size = T_train.label_size

    ckpt = tf.train.get_checkpoint_state('Model-thermal/')
    saver = tf.train.Saver()
    saver.restore(sess, 'Model-thermal/model.ckpt-14')

    datasets = ['MSRA-B', 'DUT-OMRON']




    if not os.path.exists('Result'):
        os.mkdir('Result')

    #for dataset in datasets:
        #path, imgs = load_img_list(dataset)

        #save_dir = 'Result/' + dataset
        #if not os.path.exists(save_dir):
            #os.mkdir(save_dir)

        #save_dir = 'Result/' + dataset + '/NLDF_'
        #if not os.path.exists(save_dir):
            #os.mkdir(save_dir)
    imgs_r = os.listdir('DATA/thermal-test')
    
    for f_img_r in imgs_r:

           
          

            img_r = cv2.imread(os.path.join('DATA/thermal-test', f_img_r))
            img_name, ext = os.path.splitext(f_img_r)
            

            if img_r is not None:
                #ori_img = img.copy()
                img_shape = img_r.shape
                img_r = cv2.resize(img_r, (img_size, img_size)) #- R_train.img_r_mean
                img_r = img_r.astype(np.float32) / 255.
                img_r = img_r.reshape((1, img_size, img_size, 3))

               
                start_time = time.time()
                result = sess.run(model.Prob,
                                  feed_dict={model.input_holder_t: img_r})
                print("--- %s seconds ---" % (time.time() - start_time))

                result = np.reshape(result, (label_size, label_size, 2))
                result = result[:, :, 0]

                result = cv2.resize(np.squeeze(result), (img_shape[1], img_shape[0]))

                save_name = os.path.join('Result', img_name+'.png')
                cv2.imwrite(save_name, (result*255).astype(np.uint8))

    sess.close()
