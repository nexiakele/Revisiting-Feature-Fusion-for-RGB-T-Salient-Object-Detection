#coding=utf-8
import tensorflow as tf
import vgg16
import R
import T
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
#img_r_mean=[143.711, 141.693, 139.882] #two dataset test
#img_r_mean=[144.059, 135.999, 125.807] #two dataset train
#img_r_mean=[117.361, 135.081, 150.278] #big

#img_t_mean=[85.971, 56.608, 151.944]
#img_t_mean=[101.515, 78.315, 140.606] #two dataset test
#img_t_mean=[87.832, 66.757, 135.190] #two dataset train
#img_t_mean=[83.301, 66.037, 122.097] #big

class Model:
    def __init__(self):
        self.vgg_r = R.Model()
        self.vgg_t = T.Model()
        
       
       
        self.input_holder_r = tf.placeholder(tf.float32, [1, img_size, img_size, 3])
        self.input_holder_t = tf.placeholder(tf.float32, [1, img_size, img_size, 3])
        self.label_holder = tf.placeholder(tf.float32, [label_size*label_size, 2])
        self.label_holder2 = tf.placeholder(tf.float32, [88*88, 2])
        self.label_holder3 = tf.placeholder(tf.float32, [44*44, 2])
        
      
        

        

    def build_model(self):

        #build the VGG-16 model
        vgg_r = self.vgg_r
        vgg_t = self.vgg_t
        vgg_r.build_model(self.input_holder_r)
        vgg_t.build_model(self.input_holder_t)

        #rconv1_3 = tf.nn.relu(self._dilated_conv2d(vgg_r.conv4_3, 3, 128, 1, 0.01, 'rconv1_3', biased = True))
        #rconv2_3_ = tf.nn.relu(self.Conv_2d(tf.concat([rconv2_2_, rconv2_3], axis=3),
                                           #[1, 1, 192, 128],  0.01, padding='SAME', name='rconv2_3_'))
       # rpool1_4 = tf.nn.max_pool(rconv1_3, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='rpool1_4')
        #rconv1_4 = tf.nn.relu(self._dilated_conv2d(rpool1_4, 3, 128, 3, 0.01, 'rconv1_4', biased = True))
        #rconv2_4_ = tf.nn.relu(self.Conv_2d(tf.concat([rconv2_3, rconv2_4], axis=3),
                                            #[3, 3, 128, 64],  0.01, padding='SAME', name='rconv2_4_'))
       # rpool1_5 = tf.nn.max_pool(rconv1_4, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME', name='rpool1_5')
       # rconv1_5 = tf.nn.relu(self._dilated_conv2d(rpool1_5,
                                                   #3, 64, 5, 0.01, 'rconv1_5', biased = True))
       # rpool1_6 = tf.nn.max_pool(rconv1_5, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='SAME', name='rpool1_6')
       # rconv1_6 = tf.nn.relu(self._dilated_conv2d(rpool1_6, 3, 64, 7, 0.01, 'rconv1_6', biased = True))

       # rconv1_6_ = tf.nn.relu(self.Conv_2d(tf.concat([rconv1_3, rconv1_4, rconv1_5, rconv1_6], axis=3),
                                           ### [3, 3, 384, 128],  0.01, padding='SAME', name='rconv1_6_'))


        #tconv2_2_ = tf.nn.relu(self.Conv_2d(vgg_t.conv2_2, [3, 3, 128, 128],  0.01, padding='SAME', name='tconv2_2_'))
        ##tconv1_3 = tf.nn.relu(self._dilated_conv2d(vgg_t.conv4_3, 3, 128, 1, 0.01, 'tconv1_3', biased = True))
        #tconv2_3_ = tf.nn.relu(self.Conv_2d(tf.concat([tconv2_2_, tconv2_3], axis=3),
       #                                    #[1, 1, 192, 128],  0.01, padding='SAME', name='tconv2_3_'))
       # tpool1_4 = tf.nn.max_pool(tconv1_3, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='tpool1_4')
       ### tconv1_4 = tf.nn.relu(self._dilated_conv2d(tpool1_4, 3, 128, 3, 0.01, 'tconv1_4', biased = True))
       # #rconv2_4_ = tf.nn.relu(self.Conv_2d(tf.concat([rconv2_3, rconv2_4], axis=3),
                                            #[3, 3, 128, 64],  0.01, padding='SAME', name='rconv2_4_'))
       ## tpool1_5 = tf.nn.max_pool(tconv1_4, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME', name='tpool1_5')
      #  tconv1_5 = tf.nn.relu(self._dilated_conv2d(tpool1_5,
                                                 #   3, 64, 5, 0.01, 'tconv1_5', biased = True))
       ## tpool1_6 = tf.nn.max_pool(tconv1_5, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='SAME', name='tpool1_6')
       # tconv1_6 = tf.nn.relu(self._dilated_conv2d(tpool1_6, 3, 64, 7, 0.01, 'tconv1_6', biased = True))

       # tconv1_6_ = tf.nn.relu(self.Conv_2d(tf.concat([tconv1_3, tconv1_4, tconv1_5, tconv1_6], axis=3),
                                                  #[3, 3, 384, 128],  0.01, padding='SAME', name='tconv1_6_'))
       # self.w5, self.w6 = self.attention(rconv1_6_ , tconv1_6_, name= 'fusion_conv4')
       # fusion_conv4 = self.w5*rconv1_6_+ self.w6*tconv1_6_

       # _conv4 = fusion_conv4

       
       # rconv2_3 = tf.nn.relu(self._dilated_conv2d(vgg_r.conv2_2, 3, 128, 1, 0.01, 'rconv2_3', biased = True))
        
       # rpool2_4 = tf.nn.max_pool(rconv2_3, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='rpool2_4')
        #rconv2_4 = tf.nn.relu(self._dilated_conv2d(rpool2_4, 3, 128, 3, 0.01, 'rconv2_4', biased = True))
        
       # rpool2_5 = tf.nn.max_pool(rconv2_4, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME', name='rpool2_5')
       # rconv2_5 = tf.nn.relu(self._dilated_conv2d(rpool2_5,
                                                  # 3, 64, 5, 0.01, 'rconv2_5', biased = True))
       # rpool2_6 = tf.nn.max_pool(rconv2_5, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='SAME', name='rpool2_6')
       # rconv2_6 = tf.nn.relu(self._dilated_conv2d(rpool2_6, 3, 64, 7, 0.01, 'rconv2_6', biased = True))

       # rconv2_5_ = tf.nn.relu(self.Conv_2d(tf.concat([rconv2_3, rconv2_4, rconv2_5, rconv2_6], axis=3),
                                          #  [3, 3, 384, 128],  0.01, padding='SAME', name='rconv2_5_'))


       # tconv2_3 = tf.nn.relu(self._dilated_conv2d(vgg_t.conv2_2, 3, 128, 1, 0.01, 'tconv2_3', biased = True))
        
        #tpool2_4 = tf.nn.max_pool(tconv2_3, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='tpool2_4')
        #tconv2_4 = tf.nn.relu(self._dilated_conv2d(tpool2_4, 3, 128, 3, 0.01, 'tconv2_4', biased = True))
        
       # tpool2_5 = tf.nn.max_pool(tconv2_4, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME', name='tpool2_5')
       # tconv2_5 = tf.nn.relu(self._dilated_conv2d(tpool2_5,
                                                  # 3, 64, 5, 0.01, 'tconv2_5', biased = True))
        #tpool2_6 = tf.nn.max_pool(tconv2_5, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='SAME', name='tpool2_6')
        #tconv2_6 = tf.nn.relu(self._dilated_conv2d(tpool2_6, 3, 64, 7, 0.01, 'tconv2_6', biased = True))

        #tconv2_5_ = tf.nn.relu(self.Conv_2d(tf.concat([tconv2_3, tconv2_4, tconv2_5, tconv2_6], axis=3),
                                         #   [3, 3, 384, 128],  0.01, padding='SAME', name='tconv2_5_'))

        self.w1, self.w2 = self.attention(vgg_r.rconv2_5_, vgg_t.tconv2_5_, name= 'fusion_conv2')
        self.fusion_conv2 = self.w1*vgg_r.rconv2_5_ + self.w2*vgg_t.tconv2_5_
        
        self.boundary1 = self.fusion_conv2 - tf.nn.avg_pool(self.fusion_conv2, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME')
        self.boundary2 = self.fusion_conv2 - tf.nn.avg_pool(self.fusion_conv2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

        _conv2 = self.fusion_conv2


       # rconv3_4 = tf.nn.relu(self._dilated_conv2d(vgg_r.conv3_3, 3, 128, 1, 0.01, 'rconv3_4', biased = True))
        
        
       # rpool3_5 = tf.nn.max_pool(rconv3_4, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='rpool3_5')
      #  rconv3_5 = tf.nn.relu(self._dilated_conv2d(rpool3_5, 3, 128, 3, 0.01, 'rconv3_5', biased = True))
        
       # rpool3_6 = tf.nn.max_pool(rconv3_5, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME', name='rpool3_6')
       # rconv3_6 = tf.nn.relu(self._dilated_conv2d(rpool3_6, 
                                                  # 3, 64, 5, 0.01, 'rconv3_6', biased = True))
       # rpool3_7 = tf.nn.max_pool(rconv3_6, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='SAME', name='rpool3_7')
       # rconv3_7 = tf.nn.relu(self._dilated_conv2d(rpool3_7, 3, 64, 7, 0.01, 'rconv3_7', biased = True))

        #rconv3_6_ = tf.nn.relu(self.Conv_2d(tf.concat([rconv3_4, rconv3_5, rconv3_6, rconv3_7 ], axis=3),
                                       # [3, 3, 384, 128],  0.01, padding='SAME', name='rconv3_6_'))


        

       # tconv3_4 = tf.nn.relu(self._dilated_conv2d(vgg_t.conv3_3, 3, 128, 1, 0.01, 'tconv3_4', biased = True))
        
        #tpool3_5 = tf.nn.max_pool(tconv3_4, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='tpool3_5')
        #tconv3_5 = tf.nn.relu(self._dilated_conv2d(tpool3_5, 3, 128, 3, 0.01, 'tconv3_5', biased = True))
        
        #tpool3_6 = tf.nn.max_pool(tconv3_5, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME', name='tpool3_6')    
        #tconv3_6 = tf.nn.relu(self._dilated_conv2d(tpool3_6, 
        #                                           3, 64, 5, 0.01, 'tconv3_6', biased = True))
       ## tpool3_7 = tf.nn.max_pool(tconv3_6, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='SAME', name='tpool3_7')
        #tconv3_7 = tf.nn.relu(self._dilated_conv2d(tpool3_7,
                                                     #3, 64, 7, 0.01, 'tconv3_7', biased = True))

        #tconv3_6_ = tf.nn.relu(self.Conv_2d(tf.concat([tconv3_4, tconv3_5, tconv3_6, tconv3_7], axis=3),
                                        ##[3, 3, 384, 128],  0.01, padding='SAME', name='tconv3_6_'))

        
        self.w3, self.w4 = self.attention(vgg_r.rconv3_6_, vgg_t.tconv3_6_, name= 'fusion_conv3')
        fusion_conv3 = self.w3*vgg_r.rconv3_6_  + self.w4*vgg_t.tconv3_6_ 
        
        
        
        _conv3 = fusion_conv3

        #score fusion
        #conv6_6_r = tf.nn.relu(self.Conv_2d(vgg_r.conv6_5_, [1, 1, 128, 128],  0.01, padding='SAME', name='conv6_6_r'))
        #residual_r = gate_r*vgg_r.decoder2_
 
        #conv6_6_t = tf.nn.relu(self.Conv_2d(vgg_t.conv6_5_, [1, 1, 128, 128],  0.01, padding='SAME', name='conv6_6_t'))
        #gate_t = tf.sigmoid(self.Conv_2d(vgg_t.decoder2_, [3, 3, 128, 128],  0.01, padding='SAME', name='gate_t'))
        #residual_t = gate_t*vgg_t.decoder2_

        #fusion_r = vgg_r.decoder2_ + self.upsampling_2d(conv6_6_t,name='conv6_6_t', size = (4, 4))
        #score_r_ = self.Conv_2d(fusion_r, 
                                     #[3, 3, 128, 2], 0.01, padding='SAME', name = 'score_r_' )
        #score_r_up = self.upsampling_2d(score_r_, name='score_r_up', size = (2, 2))

        #fusion_t = vgg_t.decoder2_ + self.upsampling_2d(conv6_6_r, name='conv6_6_r',size = (4, 4))
        #score_t_ = self.Conv_2d(fusion_t, 
                                     #[3, 3, 128, 2], 0.01, padding='SAME', name = 'score_t_' )
        #score_t_up = self.upsampling_2d(score_t_, name='score_t_up', size = (2, 2))

        #attention1 = tf.nn.relu(self.Conv_2d(tf.concat([fusion_r , fusion_t], axis=3),
                                            #[3, 3, 256, 256],  0.01, padding='SAME', name='attention1'))
        #attention2 = self.Conv_2d(attention1, [1, 1, 256, 2],  0.01, padding='SAME', name='attention2')
        #weight = tf.nn.softmax(attention2)
        #weight = tf.reshape(weight, [1, 176, 176, 2])    vgg_r.decoder2_ , vgg_t.decoder2_
        #w_r, w_t = tf.split(weight, 2, 3)
        #self.shape = tf.shape(w_r)
        
        #self.r = score_r_
        #self.t = score_t_
        
        #self.Score_fusion= w_r*vgg_r.score_r + w_t*vgg_t.score_t
        #self.Score_fusion= w_r*score_r_ + w_t*score_t_
        #self.Score_fusion= vgg_r.score_r + vgg_t.score_t
        #self.pre= w_r*tf.nn.softmax(vgg_r.score_r) + w_t*tf.nn.softmax(vgg_t.score_t)


        #feature fusion
        #fusion_conv2 = tf.nn.relu(self.Conv_2d(tf.concat([vgg_r.conv2_5_ , vgg_t.conv2_5_], axis=3),
                                                #[1, 1, 256, 128],  0.01, padding='SAME', name='fusion_conv2'))
        #fusion_conv2 = vgg_r.conv2_5_ + vgg_t.conv2_5_

        
        #conv6_r = self.Conv_2d(vgg_r.pool5, [1, 1, 512, 128], 0.01, padding='SAME', name = 'conv6_r' )
        #conv6_t = self.Conv_2d(vgg_t.pool5, [1, 1, 512, 128], 0.01, padding='SAME', name = 'conv6_t' )

        attention1 = self.Conv_2d(tf.concat([vgg_r.conv6_6, vgg_t.conv6_6], axis=3),
                                             [3, 3, 256, 128],  0.01, padding='SAME', name='attention1')
        attention2 = self.Conv_2d(attention1, [3, 3, 128, 2],  0.01, padding='SAME', name='attention2')
        weight = tf.nn.softmax(attention2)
        self.w_r, self.w_t = tf.split(weight, 2, 3)

        #global_fusion = w_r*vgg_r.conv6_5_ + w_t*vgg_t.conv6_5_
        #h_fusion = tf.nn.relu(self.Conv_2d(tf.concat([vgg_r.conv6_6, vgg_t.conv6_6], axis=3),
                                               # [1, 1, 256, 128],  0.01, padding='SAME', name='h_fusion'))
        #h_fusion = w_r*vgg_r.conv5_3_r + w_t*vgg_t.conv5_3_t
        
        self.h_fusion = self.w_r*vgg_r.conv6_6 + self.w_t*vgg_t.conv6_6
       # self.SA = tf.sigmoid(self.Conv_2d(self.h_fusion, [1, 1, 128, 1], 0.01, padding='SAME', name = 'SA' ))
       # SA3 = self.upsampling_2d(self.SA, name='SA1', size = (2, 2))
        #SA2 = self.upsampling_2d(self.SA, name='SA2', size = (4, 4))
        self.boundary3 = self.h_fusion - tf.nn.avg_pool(self.h_fusion, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        self.boundary4 = self.h_fusion - tf.nn.avg_pool(self.h_fusion, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME')
        self.boundary5 = self.h_fusion

       # global1_p = tf.nn.max_pool(vgg_r.conv5_3 + vgg_t.conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='global1')#conv4 = _conv4*self.SA 
       ## global1 = tf.nn.relu(self.Conv_2d(global1_p, [3, 3, 512, 128],  0.01, padding='SAME', name='global1'))
       # global2 = tf.nn.relu(self.Conv_2d(global1, [3, 3, 128, 128],  0.01, padding='SAME', name='global2'))
        #global3 = tf.nn.relu(self.Conv_2d(global1, [5, 5, 128, 128],  0.01, padding='SAME', name='global3'))
       # globalf = tf.nn.relu(self.Conv_2d(tf.concat([global2, global3], axis=3), [1, 1, 256, 128],  0.01, padding='SAME', name='globalf'))
       # global1_up = self.upsampling_2d(globalf, name='global1_up', size = (4, 4))
       # global1_up2 = self.upsampling_2d(globalf, name='global1_up', size = (8, 8))
        #h_fusion = tf.nn.relu(self.Conv_2d(tf.concat([vgg_r.conv6_6, vgg_t.conv6_6], axis=3), 
                                        # [1, 1, 256, 128],  0.01, padding='SAME', name='h_fusion'))
        #vgg_r.conv6_6 + vgg_t.conv6_6#w_r*vgg_r.conv6_6 + w_t*vgg_t.conv6_6
        #score2 = self.Conv_2d(globalf, 
                                    # [3, 3, 128, 2], 0.01, padding='SAME', name = 'score2' )
        label = tf.reshape(self.label_holder, [-1, label_size, label_size, 2])
        #label_2 = self.upsampling_2d(label, name='label_2', size = (0.0625, 0.0625))
        #score2_up = self.upsampling_2d(score2, name='score2_up')

        score_3 = self.Conv_2d(vgg_r.conv6_6, [3, 3, 128, 2], 0.01, padding='SAME', name = 'score_3' )
        label_3 = self.upsampling_2d(label, name='label_3', size = (0.125, 0.125))
        
        score_t= self.Conv_2d(vgg_t.conv6_6, [3, 3, 128, 2], 0.01, padding='SAME', name = 'score_t' )
        #lable_t = self.upsampling_2d(score_t, name='score_t_up')

        #global_fusion = w_r*score_r + w_t*score_t
        #score_fusion = self.upsampling_2d(global_fusion, name='score_fusion')
       # conv4 = _conv4*self.global_attention_layer(h_fusion, 'conv4')
        #f_decoder3 =  conv4 + h_fusion

        #f_decoder3_ = tf.nn.relu(self.Conv_2d(f_decoder3, [3, 3, 128, 128],  0.01, padding='SAME', name='f_decoder3_'))
        #f_decoder3_up = self.upsampling_2d(f_decoder3_, name='f_decoder3_up', size = (2, 2))

        #score_4 = self.Conv_2d(self.h_fusion, 
                                     #[3, 3, 128, 2], 0.01, padding='SAME', name = 'score_4' )
        #score2_up = self.upsampling_2d(score2, name='score2_up')



        global_fusion_up = self.upsampling_2d(self.h_fusion, name='global_fusion_up', size = (2, 2))
        #global_fusion_deup = tf.nn.relu(self.Deconv_2d(h_fusion, [1, 88, 88, 128], 4, 2, name='global_fusion_deup'))
        #global_fusion_up = global_fusion_biup +  global_fusion_deup

        global_fusion_up2 = self.upsampling_2d(self.h_fusion, name='global_fusion_up2', size = (4, 4))
        #global_fusion_deup2 = tf.nn.relu(self.Deconv_2d(h_fusion, [1, 176, 176, 128], 8, 4, name='global_fusion_deup2'))
        #global_fusion_up2 = global_fusion_biup2 +  global_fusion_deup2

        #global_fusion_up3 = self.upsampling_2d(h_fusion, name='global_fusion_up3')
        
        conv3 = _conv3*self.global_attention_layer(self.h_fusion, 'conv3')#tf.nn.relu(self.Conv_2d(_conv3, [3, 3, 128, 64],  0.01, padding='SAME', name='conv3'))#self.global_attention_layer(self.h_fusion, 'conv3')
        f_decoder1 = global_fusion_up + conv3#global_fusion_up + conv3


        f_decoder1_ = tf.nn.relu(self.Conv_2d(f_decoder1, [3, 3, 128, 128],  0.01, padding='SAME', name='f_decoder1_'))
        score1 = self.Conv_2d(f_decoder1_, 
                                     [3, 3, 128, 2], 0.01, padding='SAME', name = 'score1' )
        label_1 = self.upsampling_2d(label, name='label_1', size = (0.25, 0.25))
 
        #f_score_1 = self.Conv_2d(f_decoder1_, 
                                     #[3, 3, 128, 2], 0.01, padding='SAME', name = 'f_score_1' )
        #score_1 = self.upsampling_2d(f_score_1, name='score_1', size = (4, 4))

        f_decoder_up = self.upsampling_2d(f_decoder1_, name='f_decoder_up', size = (2, 2))
         
        conv2 = _conv2*self.global_attention_layer(self.h_fusion, 'conv2')#tf.nn.relu(self.Conv_2d(_conv2, [3, 3, 128, 64],  0.01, padding='SAME', name='conv2'))#self.global_attention_layer(self.h_fusion, 'conv2')
          #tf.nn.relu(self.Conv_2d(tf.concat([f_decoder_up , conv2, global_fusion_up2], axis=3),
                                        #[3, 3, 384, 128],  0.01, padding='SAME', name='f_decoder2'))
        f_decoder2 =  f_decoder_up + conv2 + global_fusion_up2

        f_decoder2_ = tf.nn.relu(self.Conv_2d(f_decoder2, [3, 3, 128, 128],  0.01, padding='SAME', name='f_decoder2_'))
       # score2 = self.Conv_2d(f_decoder2_, 
                                    # [3, 3, 128, 2], 0.01, padding='SAME', name = 'score2' )
        #score2_up = self.upsampling_2d(score2, name='score2_up', size = (2, 2))

        #f_decoder2_up = self.upsampling_2d(f_decoder2_, name='f_decoder2_up', size = (2, 2))
       # conv4 = _conv4*self.global_attention_layer(self.h_fusion, 'conv4')
        ##f_decoder3 =  f_decoder2_up + conv4
        #f_decoder3_ = tf.nn.relu(self.Conv_2d(f_decoder3, [3, 3, 128, 128],  0.01, padding='SAME', name='f_decoder3_'))
        
        #bif_score = self.upsampling_2d(f_decoder2_, name='bif_score', size = (2, 2))
        #def_score = tf.nn.relu(self.Deconv_2d(f_decoder2_, [1, 352, 352, 128], 4, 2, name='def_score'))
        #f_score = bif_score +  def_score


        f_score = self.Conv_2d(f_decoder2_, 
                                     [3, 3, 128, 2], 0.01, padding='SAME', name = 'f_score' )
        self.Score = self.upsampling_2d(f_score, name='self.Score', size = (2, 2)) 
        self.Score3 = self.upsampling_2d(score_3, name='self.Score', size = (8, 8)) 


        

        #, size = (2, 2)

        #self.Score =self.Conv_2d(tf.concat([score, score_t_up, score_t_up], axis=3),
                                                #[1, 1, 6, 2],  0.01, padding='SAME', name='Score')

        self.Score = tf.reshape(self.Score, [-1,2])
        self.Prob = tf.nn.softmax(self.Score)
        
        #self.thermal = vgg_t.Prob
       
        
        #label_holder_s = tf.image.resize_images(self.label_holder, (176, 176), method=0)
    


        self.Loss_Mean_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score1,
                                                                                  labels=label_1))
        #self.Loss_Mean_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.score_t,
                                                                                  #labels=self.label_holder)) vgg_r.Score
        self.Loss_Mean_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score_3,
                                                                                  labels=label_3))
        self.Loss_Mean_4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score_t,
                                                                                  labels=label_3))
        #self.Loss_Mean_5 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score_4,
                                                                                #  labels=label_3))self.Loss_Mean_1 + self.Loss_Mean_2 + self.Loss_Mean_3 +


        self.Loss_Mean_3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.Score,
                                                                                  labels=self.label_holder))
        self.Loss_Mean =  self.Loss_Mean_3 + 0.5*self.Loss_Mean_2 + 0.5*self.Loss_Mean_4 + 0.5*self.Loss_Mean_1#+ self.Loss_Mean_5#   self.Loss_Mean_1 + self.Loss_Mean_2 +
        #self.Loss_Mean_1 + self.Loss_Mean_2 + self.Loss_Mean_3 #+ self.Loss_Mean_4#+ self.Loss_Mean_4#+ self.C_IoU_LOSS
        #+ self.Loss_Mean_2   self.Loss_Mean_1 + + self.Loss_Mean_3
        #self.Score = tf.reshape(self.Score, [-1,2])
        self.correct_prediction = tf.equal(tf.argmax(self.Score,1), tf.argmax(self.label_holder, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


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

    