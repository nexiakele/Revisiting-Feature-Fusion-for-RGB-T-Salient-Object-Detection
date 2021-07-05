import matplotlib as mpl
mpl.use('Agg')
import cv2
import numpy as np
import RT_baseline
import os
import sys
import tensorflow as tf
import time
import vgg16
import math

import matplotlib.pyplot as plt

from pylab import *


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#img_r_mean=[143.711, 141.693, 139.882]
#img_t_mean=[101.515, 78.315, 140.606]

def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col
 
 
def visualize_feature_map(img_batch, name):
    feature_map = img_batch
    print(feature_map.shape)
 
    feature_map_combination = []
    plt.figure()
 
    num_pic = feature_map.shape[2]
    row, col = get_row_col(16)
 
    for i in range(0, 16):
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)
        plt.subplot(row, col, i + 1)
        plt.imshow(feature_map_split) #, cmap=plt.cm.summer
        axis('off')
 
    plt.savefig(os.path.join('bound', name +'_w.png'))


def load_img_list(dataset):

    if dataset == 'MSRA-B':
        path = 'dataset/MSRA-B/image'
    elif dataset == 'DUT-OMRON':
        path = 'dataset/DUT-OMRON/DUT-OMRON-image'
    
    imgs = os.listdir(path)

    return path, imgs



if __name__ == "__main__":

    model = RT_baseline.Model()
    model.build_model()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    img_size = RT_baseline.img_size
    label_size = RT_baseline.label_size

    ckpt = tf.train.get_checkpoint_state('Model_base/')
    saver = tf.train.Saver()
    saver.restore(sess, 'Model_base/model.ckpt-17')

    

    if not os.path.exists('Result-1'):
        os.mkdir('Result-1')
    

    ###m = 0
    imgs_r = os.listdir('RGB-test')
    #imgs_t = os.listdir('thermal-test')
    for f_img_r in imgs_r:
            #print(f_img_r)
            #print(f_img_t)

            path = os.path.join('RGB-test', f_img_r)
            img_r = cv2.imread(path)#.astype(np.float32)/255.0
            img_r = img_r.astype(np.float32)/255.0
            #dir, name = os.path.split(f_img_r)
            img_name, ext = os.path.splitext(f_img_r)

            f_img_t = os.path.join('thermal-test', img_name + '.png')
            img_t = cv2.imread(f_img_t)#.astype(np.float32)/255.0
            img_t = img_t.astype(np.float32)/255.0

            if img_r is not None:
                #ori_img = img.copy()
                img_shape = img_r.shape
                img_r = cv2.resize(img_r, (img_size, img_size)) #- RT.img_r_mean)
                img_r = img_r.reshape((1, img_size, img_size, 3))

                img_t = cv2.resize(img_t, (img_size, img_size)) #- RT.img_t_mean)
                img_t = img_t.reshape((1, img_size, img_size, 3))
                #, wr, wt, w1, w2, w3, w4     , model.w_r, model.w_t, model.w1, model.w2, model.w3, model.w4
                start_time = time.time()
                result = sess.run([model.Prob], 
                                 feed_dict={model.input_holder_r: img_r, model.input_holder_t: img_t})
                print("--- %s seconds ---" % (time.time() - start_time))
                #print(r) , sa   , model.SA
                #print(t) , wr, wt, w1, w2, w3, w4 , model.w_r, model.w_t, model.w1, model.w2, model.w3, model.w4]
                #f1 = np.concatenate([w1 , w2], axis=3)
               ## f1 = f1.reshape(f1.shape[1:])
                #f2 = np.concatenate([w3 , w4], axis=3)
                #f2 = f2.reshape(f2.shape[1:])
                #f3 = np.concatenate([wr , wt], axis=3)
               # f3 = f3.reshape(f3.shape[1:])
                #b = b.reshape(b.shape[1:])         ##   , b, b2, b5         , model.boundary3, model.boundary4, model.boundary5
              #  b2 = b2.reshape(b2.shape[1:])
               # b5 = b5.reshape(b5.shape[1:])
              #  visualize_feature_map(b, img_name + 'b3')
               # visualize_feature_map(b2, img_name + 'b4')
               # visualize_feature_map(b5, img_name + 'b5')

                result = np.reshape(result, (label_size, label_size, 2))###############################################################
                result = result[:, :, 0]

                result = cv2.resize(np.squeeze(result), (img_shape[1], img_shape[0]))
                #sa = cv2.resize(np.squeeze(sa), (img_shape[1], img_shape[0]))
               # wr = cv2.resize(np.squeeze(wr), (img_shape[1], img_shape[0]))
              ####  wr = wr.reshape((img_shape[1], img_shape[0], 1))
              #####  wt = cv2.resize(np.squeeze(wt), (img_shape[1], img_shape[0]))
               ### wt = wt.reshape((img_shape[1], img_shape[0], 1))
              ##  w1 = cv2.resize(np.squeeze(w1), (img_shape[1], img_shape[0]))
              #  w1 = w1.reshape((img_shape[1], img_shape[0], 1))
               # w2 = cv2.resize(np.squeeze(w2), (img_shape[1], img_shape[0]))
               ##### w2 = w2.reshape((img_shape[1], img_shape[0], 1))
               #### w3 = cv2.resize(np.squeeze(w3), (img_shape[1], img_shape[0]))
               ### w3 = w3.reshape((img_shape[1], img_shape[0], 1))
               ## w4 = cv2.resize(np.squeeze(w4), (img_shape[1], img_shape[0]))
               # w4 = w4.reshape((img_shape[1], img_shape[0], 1))
                #m = m + 1
                #f5 = np.concatenate((w1, w2, w3, w4, wr, wt), axis=2)
                #f5 = f5.reshape(f5.shape[1:])# np.squeeze(f5) 
               # visualize_feature_map(f1, img_name + 'f1')
               ## visualize_feature_map(f2, img_name + 'f2')
                #visualize_feature_map(f3, img_name + 'f3')

                save_name = os.path.join('Result-1', img_name+'.png')
                #save_sa = os.path.join('Result', img_name+'_sa.png')
                cv2.imwrite(save_name, (result*255).astype(np.uint8))
                ##cv2.imwrite(save_sa, (sa*255).astype(np.uint8))

                #save_namer = os.path.join('Result', img_name+'_wr_.png')
                #cv2.imwrite(save_namer, (wr*255).astype(np.uint8))
                #save_namet = os.path.join('Result', img_name+'_wt_.png')
                #cv2.imwrite(save_namet, (wt*255).astype(np.uint8))

                #save_name1 = os.path.join('Result', img_name+'_w1_.png')
                #cv2.imwrite(save_name1, (w1*255).astype(np.uint8))
               # save_name2 = os.path.join('Result', img_name+'_w2_.png')
                #cv2.imwrite(save_name2, (w2*255).astype(np.uint8))
               # save_name3 = os.path.join('Result', img_name+'_w3_.png')
               # cv2.imwrite(save_name3, (w3*255).astype(np.uint8))
               # save_name4 = os.path.join('Result', img_name+'_w4_.png')
                #cv2.imwrite(save_name4, (w4*255).astype(np.uint8))
                #map_t = np.reshape(map_t, (label_size, label_size, 2))
                #map_t = map_t[:, :, 0]

                #result = cv2.resize(np.squeeze(result), (img_shape[1], img_shape[0]))

                #save_name = os.path.join('map', img_name+'_RT_.png')
                #cv2.imwrite(save_name, (map_t*255).astype(np.uint8))

                #map_r = np.reshape(map_r, (label_size, label_size, 2))
                #map_r = map_r[:, :, 0]

                #result = cv2.resize(np.squeeze(result), (img_shape[1], img_shape[0]))

                #save_name = os.path.join('map_r', img_name+'_RT_.png')
                #cv2.imwrite(save_name, (map_r*255).astype(np.uint8))  , model.map_t, model.map_r  , map_t , map_r


    sess.close()
