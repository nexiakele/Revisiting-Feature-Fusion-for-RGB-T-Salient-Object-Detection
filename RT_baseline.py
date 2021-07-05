#coding=utf-8
import tensorflow as tf
import vgg16
import R_baseline
import T_baseline
import cv2
import numpy as np
from PIL import Image
import math

img_size = 384
label_size = img_size 


class Model:
    def __init__(self):
        self.vgg_r = R_baseline.Model()
        self.vgg_t = T_baseline.Model()
        
       
       
        self.input_holder_r = tf.placeholder(tf.float32, [1, img_size, img_size, 3])
        self.input_holder_t = tf.placeholder(tf.float32, [1, img_size, img_size, 3])
        self.label_holder = tf.placeholder(tf.float32, [1, label_size, label_size, 2])
        self.label_holder2 = tf.placeholder(tf.float32, [88*88, 2])
        self.label_holder3 = tf.placeholder(tf.float32, [44*44, 2])
        
      
        

        

    def build_model(self):

        #build the VGG-16 model
        vgg_r = self.vgg_r
        vgg_t = self.vgg_t
        vgg_r.build_model(self.input_holder_r)
        vgg_t.build_model(self.input_holder_t)
        
        dense1, r1_1, r1_3, r1_5, r1_7 = self.aspp(vgg_r.conv1_2, 16, 'block1')


        dense2, r2_1, r2_3, r2_5, r2_7 = self.aspp(vgg_r.conv2_2, 32, 'block2')

       
        dense3, r3_1, r3_3, r3_5, r3_7 = self.aspp(vgg_r.conv3_3, 32, 'block3')
        
    
        dense4, self.r4_1, self.r4_3, self.r4_5, self.r4_7 = self.PPM(vgg_r.conv4_3, 128, 'block4')
       

        dense5, r5_1, r5_3, r5_5, r5_7 = self.PPM(vgg_r.conv5_3, 128, 'block5')


        t_dense1, t1_1, t1_3, t1_5, t1_7 = self.aspp(vgg_t.conv1_2, 16, 't_block1')

        
        t_dense2, t2_1, t2_3, t2_5, t2_7 = self.aspp(vgg_t.conv2_2, 32, 't_block2')
        

        t_dense3, t3_1, t3_3, t3_5, t3_7 = self.aspp(vgg_t.conv3_3, 32, 't_block3')

      
        t_dense4, self.t4_1, self.t4_3, self.t4_5, self.t4_7  = self.PPM(vgg_t.conv4_3, 128, 't_block4')
        

        t_dense5, t5_1, t5_3, t5_5, t5_7 = self.PPM(vgg_t.conv5_3, 128, 't_block5')

        #f1 = tf.nn.relu(self.Conv_2d(tf.concat([vgg_r.conv1_2, vgg_t.conv1_2], axis=3), [3, 3, 128, 128],  0.01, padding='SAME', name='f1'))
        #f2 = tf.nn.relu(self.Conv_2d(tf.concat([vgg_r.conv2_2, vgg_t.conv2_2], axis=3), [3, 3, 256, 128],  0.01, padding='SAME', name='f2'))
        #f3 = tf.nn.relu(self.Conv_2d(tf.concat([vgg_r.conv3_3, vgg_t.conv3_3], axis=3), [3, 3, 512, 128],  0.01, padding='SAME', name='f3'))
        #f4 = tf.nn.relu(self.Conv_2d(tf.concat([vgg_r.conv4_3, vgg_t.conv4_3], axis=3), [1, 1, 1024, 128],  0.01, padding='SAME', name='f4'))
        #f5 = tf.nn.relu(self.Conv_2d(tf.concat([vgg_r.conv5_3, vgg_t.conv5_3], axis=3), [1, 1, 1024, 128],  0.01, padding='SAME', name='f5'))

        self.f1_1 = self.bi_crossmodel_relation_new(r1_1, t1_1, scope='fuse_block1_1')
        self.f1_3 = self.bi_crossmodel_relation_new(r1_3, t1_3, scope='fuse_block1_3')
        self.f1_5 = self.bi_crossmodel_relation_new(r1_5, t1_5, scope='fuse_block1_5')
        self.f1_7 = self.bi_crossmodel_relation_new(r1_7, t1_7, scope='fuse_block1_7')
        
        f1 = self.scale_interact_low(self.f1_1, self.f1_3, self.f1_5, self.f1_7, scope='interact_block1')

        self.f2_1 = self.bi_crossmodel_relation_new(r2_1, t2_1, scope='fuse_block2_1')
        self.f2_3 = self.bi_crossmodel_relation_new(r2_3, t2_3, scope='fuse_block2_3')
        self.f2_5 = self.bi_crossmodel_relation_new(r2_5, t2_5, scope='fuse_block2_5')
        self.f2_7 = self.bi_crossmodel_relation_new(r2_7, t2_7, scope='fuse_block2_7')

        f2 = self.scale_interact_low(self.f2_1, self.f2_3, self.f2_5, self.f2_7, scope='interact_block2')

        self.f3_1 = self.bi_crossmodel_relation_new(r3_1, t3_1, scope='fuse_block3_1')
        self.f3_3 = self.bi_crossmodel_relation_new(r3_3, t3_3, scope='fuse_block3_3')
        self.f3_5 = self.bi_crossmodel_relation_new(r3_5, t3_5, scope='fuse_block3_5')
        self.f3_7 = self.bi_crossmodel_relation_new(r3_7, t3_7, scope='fuse_block3_7')

        f3 = self.scale_interact_low(self.f3_1, self.f3_3, self.f3_5, self.f3_7, scope='interact_block3')

        self.f4_1 = self.bi_crossmodel_relation_new(self.r4_1, self.t4_1, scope='fuse_block4_1')
        self.f4_3 = self.bi_crossmodel_relation_new(self.r4_3, self.t4_3, scope='fuse_block4_3')
        self.f4_5 = self.bi_crossmodel_relation_new(self.r4_5, self.t4_5, scope='fuse_block4_5')      
        self.f4_7 = self.bi_crossmodel_relation_new(self.r4_7, self.t4_7, scope='fuse_block4_7')
        f4 = self.scale_interact_low(self.f4_1, self.f4_3, self.f4_5, self.f4_7, scope='interact_block4')

        self.f5_1 = self.bi_crossmodel_relation_new_(r5_1, t5_1, scope='fuse_block5_1')
        self.f5_3 = self.bi_crossmodel_relation_new_(r5_3, t5_3, scope='fuse_block5_3')
        self.f5_5 = self.bi_crossmodel_relation_new_(r5_5, t5_5, scope='fuse_block5_5')
        #self.f5_7 = self.bi_crossmodel_relation_new(r5_7, t5_7, scope='fuse_block5_7')

        f5 = self.scale_interact(self.f5_1, self.f5_3, self.f5_5, scope='interact_block5')

        #f5_up16 = self.upsampling_2d(f5, name='f5_up16', size = (2, 2)) 
        #node1_16 = tf.nn.relu(self.Conv_2d(f5_up16 + f4, [3, 3, 128, 128],  0.01, padding='SAME', name='node1_16'))

       # f5_up8 = self.upsampling_2d(node1_16, name='f5_up8', size = (2, 2)) 
        #node2_8 =  tf.nn.relu(self.Conv_2d(f5_up8 + f3, [3, 3, 128, 128],  0.01, padding='SAME', name='node2_8'))

        #f5_up4 = self.upsampling_2d(node2_8, name='f5_up4', size = (2, 2))
       # node3_4 = tf.nn.relu(self.Conv_2d(f5_up4 + f2, [3, 3, 128, 128],  0.01, padding='SAME', name='node3_4'))

        #f5_up2 = self.upsampling_2d(node3_4, name='f5_up2', size = (2, 2))
        #node = tf.nn.relu(self.Conv_2d(f5_up2 + f1, [3, 3, 128, 128],  0.01, padding='SAME', name='node'))

         #Iterative Deep Aggregation
        f2_up2 = self.upsampling_2d(f2, name='f2_up2', size = (2, 2))
        node1_2 =  tf.nn.relu(self.Conv_2d(tf.concat([f2_up2, f1], axis=3), [3, 3, 192, 128],  0.01, padding='SAME', name='node1_2'))

        f3_up4 = self.upsampling_2d(f3, name='f3_up4', size = (2, 2))
        node1_4 = tf.nn.relu(self.Conv_2d(tf.concat([f2, f3_up4], axis=3), [3, 3, 256, 128],  0.01, padding='SAME', name='node1_4'))
        f3_up2 = self.upsampling_2d(node1_4, name='f3_up2', size = (2, 2))
        node2_2 = tf.nn.relu(self.Conv_2d(tf.concat([node1_2, f3_up2], axis=3), [3, 3, 256, 128],  0.01, padding='SAME', name='node2_2'))
        
        f4_up8 = self.upsampling_2d(f4, name='f4_up8', size = (2, 2))
        node1_8 = tf.nn.relu(self.Conv_2d(tf.concat([f3, f4_up8], axis=3), [3, 3, 640, 256],  0.01, padding='SAME', name='node1_8'))
        f3b_ = node1_8
        f4_up4 = self.upsampling_2d(node1_8, name='f4_up4', size = (2, 2))
        node2_4 = tf.nn.relu(self.Conv_2d(tf.concat([f4_up4, node1_4], axis=3), [3, 3, 384, 192],  0.01, padding='SAME', name='node2_4'))
        f2b_ = node2_4
        f4_up2 = self.upsampling_2d(node2_4, name='f4_up2', size = (2, 2)) 
        node3_2 = tf.nn.relu(self.Conv_2d(tf.concat([f4_up2, node2_2], axis=3), [3, 3, 320, 128],  0.01, padding='SAME', name='node3_2'))
        f1b_ = node3_2



        f_score_3 = self.Conv_2d(f5, [3, 3, 384, 2],  0.01, padding='SAME', name='f_score_3')
        self.s5 = tf.nn.softmax(f_score_3)
        self.s5 = self.upsampling_2d(self.s5, name='Score5', size = (16, 16))
        label_holder_3 = self.upsampling_2d(self.label_holder, name='label_holder_3', size = (0.0625, 0.0625))

        f5_up16 = self.upsampling_2d(f5, name='f5_up16', size = (2, 2)) 
        node1_16 = tf.nn.relu(self.Conv_2d(tf.concat([f5_up16, f4], axis=3), [3, 3, 896, 256],  0.01, padding='SAME', name='node1_16'))
        f_score_1 = self.Conv_2d(node1_16, [3, 3, 256, 2],  0.01, padding='SAME', name='f_score_1')
        self.s4 = tf.nn.softmax(f_score_1)
        self.s4 = self.upsampling_2d(self.s4, name='Score4', size = (8, 8))
        label_holder_1 = self.upsampling_2d(self.label_holder, name='label_holder_1', size = (0.125, 0.125))
        lock3 = 1 - tf.reduce_max(tf.nn.softmax(f_score_1), axis=3, keepdims=True)

        f5_up8 = self.upsampling_2d(node1_16, name='f5_up8', size = (2, 2)) 
        self.lock3 = self.upsampling_2d(lock3, name='lock3_up', size = (2, 2)) 
        node2_8 =  tf.nn.relu(self.Conv_2d(tf.concat([f5_up8, f3b_], axis=3), [3, 3, 512, 192],  0.01, padding='SAME', name='node2_8'))#f3b_
        f_score_2 = self.Conv_2d(node2_8, [3, 3, 192, 2],  0.01, padding='SAME', name='f_score_2')
        self.s3 = tf.nn.softmax(f_score_2)
        self.s3 = self.upsampling_2d(self.s3, name='Score3', size = (4, 4))
        label_holder_2 = self.upsampling_2d(self.label_holder, name='label_holder_2', size = (0.25, 0.25))
        #lock2 = 1 - tf.reduce_max(tf.nn.softmax(f_score_2), axis=3, keepdims=True)
        spatial2 = self.Conv_2d(node2_8, [3, 3, 192, 1],  0.01, padding='SAME', name='spatial2')
        
        #channel2 = self.global_attention_layer(node2_8, 192, name = 'channel2')
        self.boundary2 = spatial2 - tf.nn.avg_pool(spatial2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        #foremap2 = tf.reshape(f_score_2[:,:,:,0], [-1,48,48,1])
        #self.boundary2 = tf.abs(foremap2 - tf.nn.avg_pool(foremap2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME'))
        #dilation2 = tf.nn.max_pool(tf.nn.softmax(f_score_2), ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='SAME', name='f_score_2_dia')
       # erosion2 = tf.nn.max_pool(tf.nn.softmax(f_score_2)*(-1), ksize=[1, 4, 4, 1], strides=[1, 1, 1, 1], padding='SAME', name='f_score_2_ero')*(-1)
        #ternary2 = tf.reshape((dilation2 - erosion2)[:,:,:,0], [-1,48,48,1])#tf.reduce_min((dilation2 + erosion2)/2.0, axis=3, keepdims=True)

        f5_up4 = self.upsampling_2d(node2_8, name='f5_up4', size = (2, 2)) 
       # self.lock2 = self.upsampling_2d(lock2, name='lock2_up', size = (2, 2))
        self.spatial2 = self.upsampling_2d(self.boundary2, name='spatial2_up', size = (2, 2))
        self.spatial2 = tf.sigmoid(self.spatial2)#*self.spatial2

        node3_4 = tf.nn.relu(self.Conv_2d(f5_up4 + f2b_, [3, 3, 192, 128],  0.01, padding='SAME', name='node3_4'))#tf.nn.relu(self.Conv_2d(tf.concat([f5_up4, f2b_], axis=3), [3, 3, 384, 192],  0.01, padding='SAME', name='node3_4')) #
        f_score_4 = self.Conv_2d(node3_4, [3, 3, 128, 2],  0.01, padding='SAME', name='f_score_4')
        self.s2 = tf.nn.softmax(f_score_4)
        self.s2 = self.upsampling_2d(self.s2, name='Score2', size = (2, 2))
        label_holder_4 = self.upsampling_2d(self.label_holder, name='label_holder_4', size = (0.5, 0.5))
        #lock1 = 1 - tf.reduce_max(tf.nn.softmax(f_score_4), axis=3, keepdims=True)
        spatial1 = self.Conv_2d(node3_4, [3, 3, 128, 1],  0.01, padding='SAME', name='spatial1')
        #channel1 = self.global_attention_layer(node3_4, 128, name = 'channel1')
        self.boundary1 = spatial1 - tf.nn.avg_pool(spatial1, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        
        #foremap1 = tf.reshape(f_score_4[:,:,:,0], [-1,96,96,1])
        #self.boundary1 = tf.abs(foremap1 - tf.nn.avg_pool(foremap1, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME'))
        #dilation1 = tf.nn.max_pool(tf.nn.softmax(f_score_4), ksize=[1, 11, 11, 1], strides=[1, 1, 1, 1], padding='SAME', name='f_score_4_dia')
        
        #erosion1 = tf.nn.max_pool(tf.nn.softmax(f_score_4)*(-1), ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME', name='f_score_4_ero')*(-1)
        
        #ternary1 = tf.reshape((dilation1 - erosion1)[:,:,:,0], [-1,96,96,1])#tf.reduce_min((dilation1 + erosion1)/2.0, axis=3, keepdims=True)
        
        
        f5_up2 = self.upsampling_2d(node3_4, name='f5_up2', size = (2, 2))
        #self.lock1 = self.upsampling_2d(lock1, name='lock1_up', size = (2, 2))
        self.spatial1 = self.upsampling_2d(self.boundary1, name='spatial1_up', size = (2, 2))
        self.spatial1 = tf.sigmoid(self.spatial1)#*self.spatial1
        node = tf.nn.relu(self.Conv_2d(f5_up2 + f1b_, [3, 3, 128, 128],  0.01, padding='SAME', name='node'))#tf.nn.relu(self.Conv_2d(tf.concat([f5_up2, f1b_], axis=3), [3, 3, 320, 128],  0.01, padding='SAME', name='node'))#
        f_score = self.Conv_2d(node, [3, 3, 128, 2],  0.01, padding='SAME', name='f_score') #
        

        
        self.Score = f_score#self.upsampling_2d(f_score, name='Score', size = (2, 2))
        self.Prob = tf.nn.softmax(self.Score)

        

        
        self.Loss_Mean5 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=f_score_4,
                                                                                  labels=label_holder_4))
        self.Loss_Mean4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=f_score_3,
                                                                                  labels=label_holder_3))
        self.Loss_Mean3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=f_score_1,
                                                                                  labels=label_holder_1))
        self.Loss_Mean2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=f_score_2,
                                                                                  labels=label_holder_2))
        self.Loss_Mean1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.Score,
                                                                                  labels=self.label_holder))
        self.Loss_Mean =self.loss = tf.add_n([self.Loss_Mean1, self.Loss_Mean2, self.Loss_Mean3, self.Loss_Mean4, self.Loss_Mean5]) #list summation  + regularization_losses
        Score = tf.reshape(self.Score, [-1,2])
        label_holder = tf.reshape(self.label_holder, [-1,2])
        self.correct_prediction = tf.equal(tf.argmax(Score,1), tf.argmax(label_holder, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def aspp(self, x, channels, block):
        with tf.variable_scope(block):
            #x_ = self.Conv_2d(x, [1, 1, num_x, channels*4],  0.01, padding='SAME', name='x_')
            r1 = tf.nn.relu(self._dilated_conv2d(x, 3, channels, 1, 0.01, 'r1', biased = True))
            r3 = tf.nn.relu(self._dilated_conv2d(x, 3, channels, 3, 0.01, 'r3', biased = True)) + r1
            r5 = tf.nn.relu(self._dilated_conv2d(x, 3, channels, 5, 0.01, 'r5', biased = True)) + r3
            r7 = tf.nn.relu(self._dilated_conv2d(x, 3, channels, 7, 0.01, 'r7', biased = True)) + r5
            out = tf.concat([r1, r3, r5, r7], axis=3)
            #out = self.Conv_2d(tf.concat([x, r3, r5, r7], axis=3),
                                           #[1, 1, 512, 128],  0.01, padding='SAME', name='out')
            return out, r1, r3, r5, r7

    def PPM(self, x, channels, block):
        num_x = x.shape[3].value
        with tf.variable_scope(block):
            r1 = tf.nn.relu(self.Conv_2d(x, [1, 1, num_x, channels],  0.01, padding='SAME', name='r1'))
            r3 = tf.nn.relu(self.Conv_2d(x, [3, 3, num_x, channels],  0.01, padding='SAME', name='r3'))
            r5 = tf.nn.relu(self.Conv_2d(x, [5, 5, num_x, channels],  0.01, padding='SAME', name='r5'))
            r7 = tf.nn.relu(self.Conv_2d(x, [7, 7, num_x, channels],  0.01, padding='SAME', name='r7'))#tf.nn.relu(self._dilated_conv2d(x, 4, channels, 2, 0.01, 'r7', biased = True))
            out = tf.concat([r1, r3, r5, r7], axis=3)
            #out = self.Conv_2d(tf.concat([x, r3, r5, r7], axis=3),
                                           #[1, 1, 512, 128],  0.01, padding='SAME', name='out')
            return out, r1, r3, r5, r7

    def scale_interact(self, a, b, c, scope='interact_block'):
          with tf.variable_scope(scope):
            bs, h, w, num = a.get_shape().as_list()
            a_ = tf.nn.relu(self.Conv_2d(a, [1, 1, num, num], 0.01, padding='SAME', name='a'))
            b_ = tf.nn.relu(self.Conv_2d(b, [1, 1, num, num], 0.01, padding='SAME', name='b'))
            c_ = tf.nn.relu(self.Conv_2d(c, [1, 1, num, num], 0.01, padding='SAME', name='c'))
            #d_ = tf.nn.relu(self.Conv_2d(d, [1, 1, num, num], 0.01, padding='SAME', name='d')) 
            res= tf.concat([a_, b_, c_], axis=3) #, d_
            res_1 = tf.reshape(res, shape=[-1, h*w, 3, num])
            res_2 = tf.reshape(res, shape=[-1, h*w, num, 3])
            affinity = tf.matmul(res_1, res_2)
            affinity = tf.reshape(affinity, shape=[-1, h*w, 3, 3])
            affinity = tf.nn.softmax(affinity, axis=2)
            result = tf.matmul(res_2, affinity)
            result = tf.reshape(result, shape=[-1, h, w, 3*num])
            #no = tf.concat([a, b, c, d], axis=3)
          return result

    def scale_interact_low(self, a, b, c, d, scope='interact_block'):
          with tf.variable_scope(scope):
            bs, h, w, num = a.get_shape().as_list()
            a_ = tf.reduce_mean(a, [1, 2], name='a_pool', keep_dims=True)
            a_ = tf.nn.relu(self.Conv_2d(a_, [1, 1, num, num], 0.01, padding='SAME', name='a_'))
            b_ = tf.reduce_mean(b, [1, 2], name='b_pool', keep_dims=True)
            b_ = tf.nn.relu(self.Conv_2d(b_, [1, 1, num, num], 0.01, padding='SAME', name='b_'))
            c_ = tf.reduce_mean(c, [1, 2], name='c_pool', keep_dims=True)
            c_ = tf.nn.relu(self.Conv_2d(c_, [1, 1, num, num], 0.01, padding='SAME', name='c_'))
            d_ = tf.reduce_mean(d, [1, 2], name='d_pool', keep_dims=True)
            d_ = tf.nn.relu(self.Conv_2d(d_, [1, 1, num, num], 0.01, padding='SAME', name='d_'))
            res= tf.concat([a_, b_, c_, d_], axis=3)
            res_1 = tf.reshape(res, shape=[-1, num, 4, 1])
            res_2 = tf.reshape(res, shape=[-1, num, 1, 4])
            affinity = tf.matmul(res_1, res_2)
            affinity = tf.reshape(affinity, shape=[-1, num, 4, 4])
            affinity = tf.nn.softmax(affinity, axis=2)
            res_= tf.concat([a, b, c, d], axis=3)
            res_3 = tf.reshape(res_, shape=[-1, num, h*w, 4])
            result = tf.matmul(res_3, affinity)
            result = tf.reshape(result, shape=[-1, h, w, 4*num])
            no = tf.concat([a, b, c, d], axis=3)
          return result
    
    def bi_crossmodel_relation_new(self, a, b, scope='fuse_block'):
        with tf.variable_scope(scope):
            bs, h, w, c = a.get_shape().as_list()
            a_ = tf.reduce_mean(a, [1, 2], name='a_pool', keep_dims=True)
            a_ = tf.nn.relu(self.Conv_2d(a_, [1, 1, c, c], 0.01, padding='SAME', name='a_'))
            b_ = tf.reduce_mean(b, [1, 2], name='b_pool', keep_dims=True)
            b_ = tf.nn.relu(self.Conv_2d(b_, [1, 1, c, c], 0.01, padding='SAME', name='b_'))
            
            res= tf.concat([a_, b_], axis=3)
            res_1 = tf.reshape(res, shape=[-1, c, 2, 1])
            res_2 = tf.reshape(res, shape=[-1, c, 1, 2])
            affinity = tf.matmul(res_1, res_2)
            affinity = tf.reshape(affinity, shape=[-1, c, 2, 2])
            affinity = tf.nn.softmax(affinity, axis=2)
            res_= tf.concat([a, b], axis=3)
            res_3 = tf.reshape(res_, shape=[-1, c, h*w, 2])
            result = tf.matmul(res_3, affinity)
            result = tf.reshape(result, shape=[-1, h, w, 2*c])
            fusion = self.Conv_2d(result, [1, 1, 2*c, c], 0.01, padding='SAME', name='fusion')
        return fusion

    def bi_crossmodel_relation_new_(self, a, b, scope='fuse_block'):
        with tf.variable_scope(scope):
            bs, h, w, c = a.get_shape().as_list()
            #a_ = tf.reduce_mean(a, [1, 2], name='a_pool', keep_dims=True)
            a_ = tf.nn.relu(self.Conv_2d(a, [1, 1, c, c], 0.01, padding='SAME', name='a_'))
            #b_ = tf.reduce_mean(b, [1, 2], name='b_pool', keep_dims=True)
            b_ = tf.nn.relu(self.Conv_2d(b, [1, 1, c, c], 0.01, padding='SAME', name='b_'))
            
            res= tf.concat([a_, b_], axis=3)
            res_1 = tf.reshape(res, shape=[-1, h*w, 2, c])
            res_2 = tf.reshape(res, shape=[-1, h*w, c, 2])
            affinity = tf.matmul(res_1, res_2)
            affinity = tf.reshape(affinity, shape=[-1, h*w, 2, 2])
            affinity = tf.nn.softmax(affinity, axis=2)
            res_= tf.concat([a, b], axis=3)
            res_3 = tf.reshape(res_, shape=[-1, h*w, c, 2])
            result = tf.matmul(res_3, affinity)
            result = tf.reshape(result, shape=[-1, h, w, 2*c])
            fusion = self.Conv_2d(result, [1, 1, 2*c, c], 0.01, padding='SAME', name='fusion')
            #depen = tf.reshape(affinity, shape=[-1, h, w, 4])
        return fusion

    def attention(self, input_r, input_t, name):
        with tf.variable_scope(name) as scope:
          attention1_ = self.Conv_2d(tf.concat([input_r , input_t], axis=3),
                                                  [3, 3, 256, 128],  0.01, padding='SAME', name='attention1_')
          attention2_ = self.Conv_2d(attention1_, [3, 3, 128, 2],  0.01, padding='SAME', name='attention2_')
          weight = tf.nn.softmax(attention2_)
          w_r, w_t = tf.split(weight, 2, 3)

          return w_r, w_t
            

    def Conv_2d(self, input_, shape, stddev, name, padding='SAME'):
        with tf.variable_scope(name) as scope:
            W = tf.get_variable('W',
                                shape=shape,
                                initializer=tf.truncated_normal_initializer(stddev=stddev))

            conv = tf.nn.conv2d(input_, W, [1, 1, 1, 1], padding=padding)

            b = tf.Variable(tf.constant(0.0, shape=[shape[3]]), name='b')
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
            w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o], 
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            o = tf.nn.atrous_conv2d(x, w, dilation_factor, padding='SAME')
            if biased:
                b = tf.get_variable('biases', shape=[num_o], initializer=tf.constant_initializer(0.0))
                o = tf.nn.bias_add(o, b)
            return o

    def upsampling_2d(self, tensor, name, size=(8,8)):
        h_,w_,c_ = tensor.get_shape().as_list()[1:]
        h_multi,w_multi = size
        h = tf.cast(h_multi * h_, dtype=tf.int32)
        w = tf.cast(w_multi * w_, dtype=tf.int32)
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
    def global_attention_layer(self, input_x, name):
        with tf.name_scope(name) :
             squeeze = tf.reduce_mean(input_x, [1, 2], name='squeeze', keep_dims=True)
             #Global_Average_Pooling(input_x)#net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)

             ga = self.Conv_2d(squeeze, [1, 1, 128, 128],  0.01, padding='SAME', name=name + '_ga')
             #excitation = tf.nn.relu(excitation)
             #excitation = self._fc_layer(excitation, units=out_dim, name=name+'_fully_connected2')
             excitation = tf.sigmoid(ga)

             #excitation = tf.reshape(excitation, [-1,1,1,out_dim])

             #scale = input_x * ga

             return excitation

    