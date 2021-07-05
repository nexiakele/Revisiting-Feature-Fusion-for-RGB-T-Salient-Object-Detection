#coding=utf-8
import tensorflow as tf
import vgg19
#import IAF
import cv2
import numpy as np
from PIL import Image
import math

#alpha=0.1
#beta=1
img_size = 352
label_size = img_size 
#img_r_mean=[137.160, 139.420, 138.529]
#img_r_mean=[125.810, 136.003, 144.064], 


#img_t_mean=[85.971, 56.608, 151.944]

class Model:
    def __init__(self):
        #self.vgg_r = vgg16.Vgg16()
        
       
        #self.input_holder_r = tf.placeholder(tf.float32, [1, img_size, img_size, 3])
       
        #self.label_holder = tf.placeholder(tf.float32, [label_size*label_size, 2])
    
        #self.contour_th = 1.5
        #self.contour_weight = 0.0001
        #self.sobel_fx, self.sobel_fy = self.sobel_filter()

        self.data_dict = np.load('vgg19.npy', encoding='latin1').item()
        print("R file loaded")
        print(self.data_dict.keys())

    def build_model(self, input):

        #build the VGG-16 model
        #vgg_r = self.vgg_r
        
        #vgg_r.build(self.input_holder_r)

        self.conv1_1 = self._conv_layer(input, 1, "conv1_1")
        self.conv1_2 = self._conv_layer(self.conv1_1, 1, "conv1_2")
        self.pool1 = self._max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self._conv_layer(self.pool1, 1, "conv2_1")
        self.conv2_2 = self._conv_layer(self.conv2_1, 1,"conv2_2")
        self.pool2 = self._max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self._conv_layer(self.pool2, 1, "conv3_1")
        self.conv3_2 = self._conv_layer(self.conv3_1, 1, "conv3_2")
        self.conv3_3 = self._conv_layer(self.conv3_2, 1, "conv3_3")
        self.conv3_4 = self._conv_layer(self.conv3_3, 1, "conv3_4")
        self.pool3 = self._max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self._conv_layer(self.pool3, 1, "conv4_1")
        self.conv4_2 = self._conv_layer(self.conv4_1, 1, "conv4_2")
        self.conv4_3 = self._conv_layer(self.conv4_2, 1, "conv4_3")
        self.conv4_4 = self._conv_layer(self.conv4_3, 1, "conv4_4")
        self.pool4 = self.s_max_pool(self.conv4_4, 'pool4')
        

        self.conv5_1 = self._conv_layer(self.pool4, 2, "conv5_1")
        self.conv5_2 = self._conv_layer(self.conv5_1, 2, "conv5_2")
        self.conv5_3 = self._conv_layer(self.conv5_2, 2, "conv5_3")
        self.conv5_4 = self._conv_layer(self.conv5_3, 2, "conv5_4")
        self.pool5 = self.s_max_pool(self.conv5_4, 'pool5')

        rconv2_3 = tf.nn.relu(self._dilated_conv2d(self.conv2_2, 3, 128, 1, 0.01, 'rconv2_3', biased = True))
        #rconv2_3_ = tf.nn.relu(self.Conv_2d(tf.concat([rconv2_2_, rconv2_3], axis=3),
                                           #[1, 1, 192, 128],  0.01, padding='SAME', name='rconv2_3_'))
        rpool2_4 = tf.nn.max_pool(rconv2_3, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='rpool2_4')
        rconv2_4 = tf.nn.relu(self._dilated_conv2d(rpool2_4, 3, 128, 3, 0.01, 'rconv2_4', biased = True))
        #rconv2_4_ = tf.nn.relu(self.Conv_2d(tf.concat([rconv2_3, rconv2_4], axis=3),
                                            #[3, 3, 128, 64],  0.01, padding='SAME', name='rconv2_4_'))
        rpool2_5 = tf.nn.max_pool(rconv2_4, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME', name='rpool2_5')
        rconv2_5 = tf.nn.relu(self._dilated_conv2d(rpool2_5,
                                                   3, 64, 5, 0.01, 'rconv2_5', biased = True))
        rpool2_6 = tf.nn.max_pool(rconv2_5, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='SAME', name='rpool2_6')
        rconv2_6 = tf.nn.relu(self._dilated_conv2d(rpool2_6, 3, 64, 7, 0.01, 'rconv2_6', biased = True))

        self.rconv2_5_ = tf.nn.relu(self.Conv_2d(tf.concat([rconv2_3, rconv2_4, rconv2_5, rconv2_6], axis=3),
                                            [3, 3, 384, 128],  0.01, padding='SAME', name='rconv2_5_'))
    
        
        rconv3_4 = tf.nn.relu(self._dilated_conv2d(self.conv3_4, 3, 128, 1, 0.01, 'rconv3_4', biased = True))
        #rconv3_4_ = tf.nn.relu(self.Conv_2d(tf.concat([rconv3_3_ , rconv3_4], axis=3),
                                           #[1, 1, 192, 128],  0.01, padding='SAME', name='rconv3_4_'))
        
        rpool3_5 = tf.nn.max_pool(rconv3_4, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='rpool3_5')
        rconv3_5 = tf.nn.relu(self._dilated_conv2d(rpool3_5, 3, 128, 3, 0.01, 'rconv3_5', biased = True))
        #conv3_5_ = tf.nn.relu(self.Conv_2d(tf.concat([conv3_4, conv3_5], axis=3),
                                            #[1, 1, 128, 64],  0.01, padding='SAME', name='conv3_5_'))
        rpool3_6 = tf.nn.max_pool(rconv3_5, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME', name='rpool3_6')
        rconv3_6 = tf.nn.relu(self._dilated_conv2d(rpool3_6, 
                                                   3, 64, 5, 0.01, 'rconv3_6', biased = True))
        rpool3_7 = tf.nn.max_pool(rconv3_6, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='SAME', name='rpool3_7')
        rconv3_7 = tf.nn.relu(self._dilated_conv2d(rpool3_7, 3, 64, 7, 0.01, 'rconv3_7', biased = True))

        self.rconv3_6_ = tf.nn.relu(self.Conv_2d(tf.concat([rconv3_4, rconv3_5, rconv3_6, rconv3_7 ], axis=3),
                                        [3, 3, 384, 128],  0.01, padding='SAME', name='rconv3_6_'))

        #pool5_ = self._max_pool(pool4_s, 'pool5_')
        #pool5_s  = tf.nn.relu(self.Conv_2d(pool5_, [3, 3, 512, 512],  0.01, padding='SAME', name='pool5_s'))
        #pool5_1 = tf.nn.relu(self.Conv_2d(pool5_s, [5, 5, 512, 128],  0.01, padding='SAME', name='pool5_1'))
        #pool5_up = self.upsampling_2d(pool5_1, name='pool5_up', size = (4, 4))
       
        
    
        self.conv5_3_r = tf.nn.relu(self.Conv_2d(self.pool5, [3, 3, 512, 256],  0.01, padding='SAME', name='conv5_3_r'))
        

        #conv5_3_ = tf.nn.relu(self.Conv_2d(tf.concat([conv5_3_r, pool4_up], axis=3),
                                           #[1, 1, 256, 128],  0.01, padding='SAME', name='conv5_3_'))

        #conv6_1 = tf.nn.relu(self._dilated_conv2d(conv5_3_r, 3, 128, 1, 0.01, 'conv6_1', biased = True))
        #conv6_1_ = tf.nn.relu(self.Conv_2d(tf.concat([conv5_3_r, conv6_1], axis=3),
                                          #[1, 1, 192, 128],  0.01, padding='SAME', name='conv6_1_'))
        #pool6_2 = tf.nn.max_pool(self.conv5_3_r, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], 
                                  #padding='SAME', name='pool6_2')
        rconv6_2 = tf.nn.relu(self._dilated_conv2d(self.conv5_3_r, 3, 128, 2, 0.01, 'rconv6_2', biased = True))

        #conv6_2_ = tf.nn.relu(self.Conv_2d(tf.concat([self.conv5_3_r, conv6_2], axis=3),
                                            #[3, 3, 384, 256],  0.01, padding='SAME', name='conv6_2_'))
        #pool6_3 = tf.nn.avg_pool(conv6_2, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME')
        pool6_3 = tf.nn.max_pool(rconv6_2, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool6_3')
        rconv6_3 = tf.nn.relu(self._dilated_conv2d(pool6_3, 3, 128, 5, 0.01, 'rconv6_3', biased = True))
        rconv6_3_ = tf.nn.relu(self.Conv_2d(tf.concat([self.conv5_3_r, rconv6_3], axis=3), [1, 1, 384, 256],  0.01, padding='SAME', name='rconv6_3_'))

        #pool6_4 = tf.nn.avg_pool(conv6_3, ksize=[1, 9, 9, 1], strides=[1, 1, 1, 1], padding='SAME'
        pool6_4 = tf.nn.max_pool(rconv6_3_, ksize=[1, 9, 9, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool6_4')
        rconv6_4 = tf.nn.relu(self._dilated_conv2d(pool6_4, 3, 128, 9, 0.01, 'rconv6_4', biased = True))
        #conv6_4_ = tf.nn.relu(self.Conv_2d(conv6_4,
                               #[1, 1, 128, 64],  0.01, padding='SAME', name='conv6_4_'))
        #pool6_5 = tf.nn.avg_pool(conv6_4, ksize=[1, 15, 15, 1], strides=[1, 1, 1, 1], padding='SAME')
        pool6_5 = tf.nn.max_pool(rconv6_4, ksize=[1, 15, 15, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool6_5')
        rconv6_5 = tf.nn.relu(self._dilated_conv2d(pool6_5, 3, 128, 15, 0.01, 'rconv6_5', biased = True))
        #conv6_5_ = tf.nn.relu(self.Conv_2d(conv6_5,
                                #[1, 1, 128, 64],  0.01, padding='SAME', name='conv6_5_'))
        
        self.rconv6_6 = tf.nn.relu(self.Conv_2d(tf.concat([self.conv5_3_r, rconv6_2, rconv6_3, rconv6_4, rconv6_5], axis=3),
                                                                      [1, 1, 768, 128],  0.01, padding='SAME', name='rconv6_6'))
        #self.score3 = self.Conv_2d(conv6_5_ , 
                                     #[1, 1, 128, 2], 0.01, padding='SAME', name = 'score3' )

        #conv6_up = self.upsampling_2d(conv6_5_, name='conv6_up', size = (2, 2))
        #conv6_up2 = self.upsampling_2d(conv6_5_, name='conv6_up2', size = (4, 4))

        #decoder1 = conv6_up + conv3
        #decoder1 = self.Conv_2d(tf.concat([conv6_up, conv3], axis=3),
                                          #[1, 1, 320, 128],  0.01, padding='SAME', name='decoder1')
        #self.decoder1_ = tf.nn.relu(self.Conv_2d(decoder1, [3, 3, 128, 128],  0.01, padding='SAME', name='decoder1_'))
        #decoder1_up = self.Conv_2d(decoder1_, [1, 1, 128, 256],  0.01, padding='SAME', name='decoder1_up')
        #self.score2 = self.Conv_2d(decoder1_, 
                                     #[3, 3, 128, 2], 0.01, padding='SAME', name = 'score2' )

        #decoder_up = self.upsampling_2d(self.decoder1_, name='decoder_up', size = (2, 2))
 
        #decoder2 = decoder_up + conv2 + conv6_up2 #tf.nn.relu()
        #decoder2 = self.Conv_2d(tf.concat([conv6_up, conv2], axis=3),
                                          #[1, 1, 256, 128],  0.01, padding='SAME', name='decoder2')
        #self.decoder2_ = tf.nn.relu(self.Conv_2d(decoder2, [3, 3, 128, 128],  0.01, padding='SAME', name='decoder2_'))
        #decoder2_up = self.Conv_2d(decoder2_, [1, 1, 128, 256],  0.01, padding='SAME', name='decoder2_up')

        #conv6_5 = tf.nn.relu(self._dilated_conv2d(conv6_4_ , 3, 64, 13, 0.01, 'conv6_5', biased = True))
        #conv6_5_ = self.Conv_2d(tf.concat([conv5_3_r, conv6_1, conv6_2, conv6_3, conv6_4, conv6_5], axis=3),
                                           #[1, 1, 448, 128],  0.01, padding='SAME', name='conv6_5_')
        #self.decoder3 = self.Conv_2d(self.conv6_5_, [3, 3, 128, 128],  0.01, padding='SAME', name='decoder3')#self.decoder2_ + conv6_up2
        #self.score_r = self.Conv_2d(self.conv6_6, 
                                   #  [3, 3, 128, 2], 0.01, padding='SAME', name = 'score_r' )
        #self.score_t = tf.nn.relu(self.Deconv_2d(self.score_t_up, [1, 128, 128, 2], 4, 4, name='score_t'))
        #self.score_t = self.upsampling_2d(self.score_t_up, name='score_t', size = (4, 4))


    
        #self.Score = self.upsampling_2d(self.score_r, name='Score') #, size = (2, 2)

        
        #score fusion
        
      
        #self.Score = tf.reshape(self.Score, [-1,2])
        #self.Prob = tf.nn.softmax(self.Score)

        #Get the contour term
        #self.Prob_C = tf.reshape(self.Prob, [1, 256, 256, 2])
        #self.Prob_Grad = tf.tanh(self.im_gradient(self.Prob_C))
        #self.Prob_Grad = tf.tanh(tf.reduce_sum(self.im_gradient(self.Prob_C), reduction_indices=3, keep_dims=True))

        #self.label_C = tf.reshape(self.label_holder, [1, 256, 256, 2])
        #self.label_Grad = tf.cast(tf.greater(self.im_gradient(self.label_C), self.contour_th), tf.float32)
        #self.label_Grad = tf.cast(tf.greater(tf.reduce_sum(self.im_gradient(self.label_C),
                                                           #reduction_indices=3, keep_dims=True),
                                             #self.contour_th), tf.float32)

        #self.C_IoU_LOSS = self.Loss_IoU(self.Prob_Grad, self.label_Grad)


       

        #self.Loss_Mean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.Score,
                                                                                  #labels=self.label_holder))
        #self.Loss = self.Loss_Mean + self.C_IoU_LOSS
        #
        #self.Score = tf.reshape(self.Score, [-1,2])
        #self.correct_prediction = tf.equal(tf.argmax(self.Score,1), tf.argmax(self.label_holder, 1))
        #self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def Conv_2d(self, input_, shape, stddev, name, padding='SAME'):
        with tf.variable_scope(name) as scope:
            #W = tf.Variable(self.data_dict[name + '/W'], name="weights")#self.get_conv_filter(name)
            W = tf.get_variable('W',
                                shape=shape,
                                initializer=tf.truncated_normal_initializer(stddev=stddev))

            conv = tf.nn.conv2d(input_, W, [1, 1, 1, 1], padding=padding)

            b = tf.Variable(tf.constant(0.0, shape=[shape[3]]), name='b')
            #b = tf.Variable(self.data_dict[name + '/b'], name="biases")#self.get_bias(name)
            conv = tf.nn.bias_add(conv, b)

            return conv

    def Conv_2d_down(self, input_, shape, stddev, name, padding='SAME'):
        with tf.variable_scope(name) as scope:
            W = tf.get_variable('W',
                                shape=shape,
                                initializer=tf.truncated_normal_initializer(stddev=stddev))

            conv = tf.nn.conv2d(input_, W, [1, 2, 2, 1], padding=padding)

            b = tf.Variable(tf.constant(0.0, shape=[shape[3]]), name='b')
            conv = tf.nn.bias_add(conv, b)

            return conv

    def Deconv_2d(self, input_, output_shape,
                  k_s=3, st_s=2, stddev=0.01, padding='SAME', name="deconv2d"):
        with tf.variable_scope(name):
            W = tf.get_variable('W',
                                shape=[k_s, k_s, output_shape[3], input_.get_shape()[3]],
                                initializer=tf.random_normal_initializer(stddev=stddev))

            deconv = tf.nn.conv2d_transpose(input_, W, output_shape=output_shape,
                                            strides=[1, st_s, st_s, 1], padding=padding)

            b = tf.get_variable('b', [output_shape[3]], initializer=tf.constant_initializer(0.0))
            deconv = tf.nn.bias_add(deconv, b)

        return deconv

    def _dilated_conv2d(self, x, kernel_size, num_o, dilation_factor, stddev ,name, biased=False):
        """
        Dilated conv2d without BN or relu.
        """
        num_x = x.shape[3].value
        with tf.variable_scope(name) as scope:
            #w = tf.Variable(self.data_dict[name + '/weights'], name="weights")#self.get_conv_filter(name)
            w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o], 
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            o = tf.nn.atrous_conv2d(x, w, dilation_factor, padding='SAME')
            if biased:
                #b = tf.Variable(self.data_dict[name + '/biases'], name="biases")#self.get_bias(name)
                b = tf.get_variable('biases', shape=[num_o], initializer=tf.constant_initializer(0.0))
                o = tf.nn.bias_add(o, b)
            return o

    def upsampling_2d(self, tensor, name, size=(8,8)):
        h_,w_,c_ = tensor.get_shape().as_list()[1:]
        h_multi,w_multi = size
        h = h_multi * h_
        w = w_multi * w_
        target = tf.image.resize_bilinear(tensor,size=(h,w),name='deconv_{}'.format(name))

        return target

    def sobel_filter(self):
        fx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)
        fy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).astype(np.float32)

        fx = np.stack((fx, fx), axis=2)
        fy = np.stack((fy, fy), axis=2)

        fx = np.reshape(fx, (3, 3, 2, 1))
        fy = np.reshape(fy, (3, 3, 2, 1))

        tf_fx = tf.Variable(tf.constant(fx))
        tf_fy = tf.Variable(tf.constant(fy))

        return tf_fx, tf_fy

    def im_gradient(self, im):
        gx = tf.nn.depthwise_conv2d(tf.pad(im, [[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC'),
                                    self.sobel_fx, [1, 1, 1, 1], padding='VALID')
        gy = tf.nn.depthwise_conv2d(tf.pad(im, [[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC'),
                                    self.sobel_fy, [1, 1, 1, 1], padding='VALID')
        return tf.sqrt(tf.add(tf.square(gx), tf.square(gy)))

    def Loss_IoU(self, pred, gt):
        inter = tf.reduce_sum(tf.multiply(pred, gt))
        union = tf.add(tf.reduce_sum(tf.square(pred)), tf.reduce_sum(tf.square(gt)))

        if inter == 0:
            return 0
        else:
            return 1 - (2*(inter+1)/(union + 1))

    def L2(self, tensor, wd=0.0005):
        return tf.mul(tf.nn.l2_loss(tensor), wd, name='L2-Loss')

    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

    def s_max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1],
                              padding='SAME', name=name)

    def _conv_layer(self, bottom, rate, name):
        with tf.variable_scope(name) as scope:
            filt = self.get_conv_filter(name)#tf.Variable(self.data_dict[name + '/filter'], name="weights")#self.get_conv_filter(name)
            #conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv = tf.nn.atrous_conv2d(bottom, filt, rate, padding = 'SAME')

            conv_biases = self.get_bias(name)#tf.Variable(self.data_dict[name + '/biases'], name="biases")#
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def get_conv_filter(self, name):

        return tf.Variable(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.Variable(self.data_dict[name][1], name="biases")
