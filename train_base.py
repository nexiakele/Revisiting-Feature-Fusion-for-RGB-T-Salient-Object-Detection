#coding=utf-8
import cv2
import numpy as np
#import model
import RT_baseline
import vgg16
import tensorflow as tf
import os
from setproctitle import setproctitle
setproctitle('不辣')
#import importlib,sys
#importlib.reload(sys)

#learning_rate = 1e-7 
#decay_rate = 0.96  
#decay_steps = 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#def load_training_list():

    #with open('dataset/MSRA-B/train1.txt') as f:
        #lines = f.read().splitlines()

    #files = []
    #labels = []

    #for line in lines:
        #labels.append('%s' % line)
        #files.append('%s' % line.replace('.png', '.jpg'))

    #return files, labels


def load_train_list():

    files_r = []

    with open('/home/ly/disk2/xiao/RGBT1/DATA/train_r1.txt') as f:

        lines = f.read().splitlines()

    for line in lines:
        files_r.append('%s' % line)
        #files.append('%s' % line.replace('.png', '.jpg'))


    

    return files_r


if __name__ == "__main__":
    #sess = tf.Session()
    model = RT_baseline.Model()
    model.build_model()

    sess = tf.Session()
    
    #Prob_C = tf.reshape(model.Prob, [1, 256, 256, 2])
    #Prob_Grad = tf.tanh(model.im_gradient(Prob_C))
    #Prob_Grad = tf.tanh(tf.reduce_sum(model.im_gradient(Prob_C), reduction_indices=3, keep_dims=True))

    #label_C = tf.reshape(model.label_holder, [1, 256, 256, 2])
    #label_Grad = tf.cast(tf.greater(model.im_gradient(label_C), model.contour_th), tf.float32)
    #label_Grad = tf.cast(tf.greater(tf.reduce_sum(model.im_gradient(label_C),
                                                           #reduction_indices=3, keep_dims=True),
                                             #model.contour_th), tf.float32)

    #C_IoU_LOSS = model.Loss_IoU(Prob_Grad, label_Grad)
    #global_step = tf.Variable(0, dtype=tf.int64, name='global_step')

    #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    #learning_rate = tf.train.exponential_decay(1e-4, global_step,
                                               #1000, 0.9, staircase=True)

    max_grad_norm = 1
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(model.Loss_Mean , tvars), max_grad_norm)
    
    #global_ = tf.Variable(tf.constant(0))  
    #c = tf.train.exponential_decay(learning_rate, global_, decay_steps, decay_rate, staircase=True)
    opt = tf.train.AdamOptimizer(1e-5) 
    #_stop = tf.stop_gradient(model.logits)
       

    train_op = opt.apply_gradients(zip(grads, tvars)) #, global_step = global_step
    

    sess.run(tf.global_variables_initializer())

    #train_list_r_s = load_train_list()

    train_list_r=os.listdir('/home/ly/disk2/xiao/RGBT2/DATA/RGB')#[]
    
    #indexs = [i for i in range(len(train_list_r_s))]
   # np.random.shuffle(indexs)
    #for index in indexs:
        #train_list_r.append( train_list_r_s[index])
        
    saver = tf.train .Saver(max_to_keep=20)
    n_epochs = 25
    img_size = RT_baseline.img_size
    label_size = RT_baseline.label_size
    if not os.path.exists('Model_base'):
        os.mkdir('Model_base')
    
    ckpt = tf.train.get_checkpoint_state('Model_base/')   
    if ckpt and ckpt.all_model_checkpoint_paths:  
        saver.restore(sess,ckpt.model_checkpoint_path)
        #saver.restore(sess,'Model/model.ckpt-4')  
    #else:
        #os.mkdir('Model')
    # for i in xrange(n_epochs):
    for i in range(n_epochs):

        whole_loss = 0.0
        whole_acc = 0.0

        count = 0
        for f_img_r in train_list_r:
            #print(f_img_r)
            #print(f_img_t)

            #print(f_label)
            #dirname,filename=os.path.split(f_img_r)
            #name, ext = os.path.splitext(filename)

            #thermal = cv2.imread(os.path.join('thermal', name + '.png'))
            #gt = cv2.imread(os.path.join('GT', name + '.png'))

            
            f_img_r = os.path.join('/home/ly/disk2/xiao/RGBT2/DATA/RGB', f_img_r)
            img_r = cv2.imread(f_img_r).astype(np.float32)/255.0
           
            dirname,filename=os.path.split(f_img_r)
            name, ext = os.path.splitext(filename)

            #thermal = cv2.imread(os.path.join('thermal', name + '.png'))
            f_img_t = os.path.join('/home/ly/disk2/xiao/RGBT2/DATA/thermal', name + '.png')

            
            img_t = cv2.imread(f_img_t).astype(np.float32)/255.0
            #img_90_t = np.rot90(img_t)
            #img_180_t = np.rot90(img_90_t)
            #img_270_t = np.rot90(img_180_t)

            # print(img)
            #img_flip_t = cv2.flip(img_t, 1)
            #img_flip_90_t = np.rot90(img_flip_t)
            #img_flip_180_t = np.rot90(img_flip_90_t)
            #img_flip_270_t = np.rot90(img_flip_180_t)
            #res4 = model.image_entropy(img_flip_t)
            f_label = os.path.join('/home/ly/disk2/xiao/RGBT2/DATA/GT', name + '.png')

            label = cv2.imread(f_label)[:, :, 0].astype(np.float32)
            #label_90 = np.rot90(label)
            #label_180 = np.rot90(label_90)
            #label_270 = np.rot90(label_180)

            #label_flip = cv2.flip(label, 1)
            #label_flip_90 = np.rot90(label_flip)
            #label_flip_180 = np.rot90(label_flip_90)
            #label_flip_270 = np.rot90(label_flip_180)
            


            img_r = cv2.resize(img_r, (img_size, img_size)) #- RT.img_r_mean)
            img_t = cv2.resize(img_t, (img_size, img_size)) #- RT.img_t_mean)
            label = cv2.resize(label, (label_size, label_size))
            label = label.astype(np.float32) / 255.
            

            img_r = img_r.reshape((1, img_size, img_size, 3))
            img_t = img_t.reshape((1, img_size, img_size, 3))
            
            label = np.stack((label, 1-label), axis=2)
            label = label.reshape((1, label_size, label_size, 2))
            

            #label = np.reshape(label, [-1, 2])
          
             
            #weight = sess.run([model.w], feed_dict={model.input_holder_r: img_r})
            #temp = weight.eval()
             
            #res3 = temp[0][0] ]   , acc    , model.accuracy
            _, loss, acc= sess.run([train_op, model.Loss_Mean, model.accuracy],
                                          feed_dict={model.input_holder_r: img_r, model.input_holder_t: img_t, 
                                                     model.label_holder: label})

           # _, loss, acc, rate = sess.run([train_op, model.Loss_Mean, model.accuracy, learning_rate],
                                          #feed_dict={model.input_holder_r: img_r, model.input_holder_t: img_t, 
                                              # model.label_holder: label})

            whole_loss += loss
            whole_acc += acc
            count = count + 1

            # add horizon flip image for training
            #img_flip_r = cv2.resize(img_flip_r, (img_size, img_size)) - RT.img_r_mean
            #img_flip_t = cv2.resize(img_flip_t, (img_size, img_size)) - RT.img_t_mean
            #label_flip = cv2.resize(label_flip, (label_size, label_size))
            #label_flip = label_flip.astype(np.float32) / 255.

            #img_flip_r = img_flip_r.reshape((1, img_size, img_size, 3))
            #img_flip_t = img_flip_t.reshape((1, img_size, img_size, 3))
            #label_flip = np.stack((label_flip, 1 - label_flip), axis=2)
            #label_flip = np.reshape(label_flip, [-1, 2])

            #_, loss, acc = sess.run([train_op, model.Loss_Mean, model.accuracy],
                                    #feed_dict={model.input_holder_r: img_flip_r, model.input_holder_t: img_flip_t, model.w1: res1,
                                               #model.label_holder: label_flip, model.w2: res2,})

            #whole_loss += loss
            #whole_acc += acc
            #count = count + 1
             
            

            if count % 100 == 0:
                print("Loss of %d images: %f, Accuracy: %f" % (count, (whole_loss/count), (whole_acc/count)))
                #print(shape)   , Accuracy: %f  , (whole_acc/count)

        print("Epoch %d: %f" % (i, (whole_loss/len(train_list_r))))

        saver.save(sess, 'Model_base/model.ckpt', global_step=i)

    sess.close()
