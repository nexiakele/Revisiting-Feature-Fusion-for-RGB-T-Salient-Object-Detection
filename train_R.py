#coding=utf-8
import cv2
import numpy as np
#import model
import R_train
import vgg16
import tensorflow as tf
import os

#import importlib,sys
#importlib.reload(sys)

#learning_rate = 1e-5
#decay_rate = 0.93 
#decay_steps = 2500
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_train_list():

    files_r = []
    #files_t = []
    #labels = []

    with open('/home/ly/disk2/xiao/RGBT/datas/train_r.txt') as f:

        lines = f.read().splitlines()

    for line in lines:
        files_r.append('%s' % line)
    


    #with open('/home/ly/disk2/xiao/RGBT/gt.txt') as f:
        #lines = f.read().splitlines()

    #for line in lines:
       # labels.append('%s' % line)
       

    return files_r


if __name__ == "__main__":
    #sess = tf.Session()
    model = R_train.Model()
    model.build_model()

    sess = tf.Session()
    #sobel_fx, sobel_fy = sobel_filter()
    #Prob_C = tf.reshape(model.Prob, [1, 256, 256, 2])
    #Prob_Grad = tf.tanh(im_gradient(Prob_C))
    #Prob_Grad = tf.tanh(tf.reduce_sum(im_gradient(Prob_C), reduction_indices=3, keep_dims=True))

    #label_C = tf.reshape(model.label_holder, [1, 256, 256, 2])
    #label_Grad = tf.cast(tf.greater(im_gradient(label_C), contour_th), tf.float32)
    #label_Grad = tf.cast(tf.greater(tf.reduce_sum(im_gradient(label_C),
                                                           #reduction_indices=3, keep_dims=True),
                                             #contour_th), tf.float32)

    #C_IoU_LOSS = Loss_IoU(Prob_Grad, label_Grad)#loss_IOU(model.Score, model.label_holder)
    Loss_Mean1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model.Score, labels=model.label_holder))
    Loss_Mean2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model.score3, labels=model.label3))
    Loss_Mean3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model.score2, labels=model.label2))
    #Loss_Mean2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model.score_dsn1_up, labels=model.label_holder))
    #Loss_Mean3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model.score_dsn2_up, labels=model.label_holder))
    #Loss_Mean4 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model.score_dsn3_up, labels=model.label_holder))
    #Loss_Mean5 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model.score_dsn4_up, labels=model.label_holder))
    #Loss_Mean6 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model.score_dsn5_up, labels=model.label_holder))
    #Loss_Mean7 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model.score_dsn6_up, labels=model.label_holder))
    Loss = Loss_Mean1 + Loss_Mean2 + Loss_Mean3 # + model.C_IoU_LOSS #+ Loss_Mean2 + Loss_Mean3 + Loss_Mean4 + Loss_Mean5 + Loss_Mean6 + Loss_Mean7#contour_weight = 0.0001 C_IoU_LOSS + 

    max_grad_norm = 1
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(Loss, tvars), max_grad_norm)
    
    #global_step = tf.Variable(0, dtype=tf.int64, name='global_step')
    #learning_rate = tf.train.exponential_decay(1e-5, global_step,
                                               #2500, 0.93, staircase=True)
    opt = tf.train.AdamOptimizer(1e-5)
    #_stop = tf.stop_gradient(model.logits)
       

    train_op = opt.apply_gradients(zip(grads, tvars)) #, global_step = global_step
    

    sess.run(tf.global_variables_initializer())

    train_list_r_s = load_train_list()

    train_list_r=[]
   
    #label_list=[]
    indexs = [i for i in range(len(train_list_r_s))]
    np.random.shuffle(indexs)
    for index in indexs:
        train_list_r.append( train_list_r_s[index])
       
        #label_list.append( label_list_s[index])

    saver = tf.train.Saver(max_to_keep=7)
    n_epochs = 15
    img_size = R_train.img_size
    label_size = R_train.label_size
    if not os.path.exists('Model'):
        os.mkdir('Model')
    
    ckpt = tf.train.get_checkpoint_state('Model/')   
    if ckpt and ckpt.all_model_checkpoint_paths:  
        saver.restore(sess,ckpt.model_checkpoint_path)
        #saver.restore(sess,'Model/model.ckpt-30')  
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

            

            img_r = cv2.imread(f_img_r).astype(np.float32)/255.0
            
            dirname,filename=os.path.split(f_img_r)
            name, ext = os.path.splitext(filename)

            #thermal = cv2.imread(os.path.join('thermal', name + '.png'))
            f_label = os.path.join('datas/GT', name + '.png')
            #print(f_label)

            label = cv2.imread(f_label)[:, :, 0].astype(np.float32)
           


            img_r = cv2.resize(img_r, (img_size, img_size)) #- R_train.img_r_mean
            
            label = cv2.resize(label, (label_size, label_size))
            label = label.astype(np.float32) / 255.
            #label2 = cv2.resize(label, (88, 88))
            #label2 = label2.astype(np.float32) / 255.
            #label3 = cv2.resize(label, (44, 44))
            #label3 = label3.astype(np.float32) / 255.


            img_r = img_r.reshape((1, img_size, img_size, 3))
            #label = label.reshape((1, label_size, label_size, 1))
          
            label = np.stack((label, 1-label), axis=2)
            #label2 = np.stack((label2, 1-label2), axis=2)
           # label3 = np.stack((label3, 1-label3), axis=2)

            label = np.reshape(label, [-1, 2])
            #label2 = np.reshape(label2, [-1, 2])
            #label3 = np.reshape(label3, [-1, 2])
             
            #  
            _, loss , acc= sess.run([train_op, Loss, model.accuracy],
                                      feed_dict={model.input_holder_r: img_r, model.label_holder: label})

            whole_loss += loss
            whole_acc += acc
            count = count + 1
             
            

            if count % 100 == 0:
                print("Loss of %d images: %f, Accuracy: %f" % (count, (whole_loss/count), (whole_acc/count)))#ss

        print("Epoch %d: %f" % (i, (whole_loss/len(train_list_r))))
        #print(rate) #rate = learning_rate

        saver.save(sess, 'Model/model.ckpt', global_step=i)
