#coding=utf-8
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

checkpoint_path='Model-RGB/model.ckpt-14'#your ckpt path
reader=pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map=reader.get_variable_to_shape_map()

R={}
R_layer = ['conv1_1', 'conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3','conv4_1','conv4_2','conv4_3',
               'conv5_1','conv5_2','conv5_3','conv2_3','conv2_3_','conv2_4','conv2_5','conv2_5_',
               'gate2','conv3_3_','conv3_4','conv3_4_', 'conv3_5', 'conv3_6', 'conv3_6_', 'gate3', 'conv5_3_t',
               'conv6_1', 'conv6_1_', 'conv6_2', 'conv6_2_', 'conv6_3', 'conv6_4','conv6_5','conv6_4_', 'decoder1_', 
               'decoder2_', 'score_t','conv6_6']
#add_info = ['filter','weights','W','b','biases']

#R={'conv1_1':[[],[]],'conv1_2':[[],[]],'conv2_1':[[],[]],'conv2_2':[[],[]], 'conv3_1':[[],[]],'conv3_2':[[],[]],'conv3_3':[[],[]],'conv4_1':[[],[]],
   #  'conv4_2':[[],[]],'conv4_3':[[],[]],'conv5_1':[[],[]],'conv5_2':[[],[]], 'conv5_3':[[],[]], 'Fea_Global_1':[[],[]], 
   ##   'Fea_Global_2':[[],[]], 'Fea_Global':[[],[]], 'Fea_P5':[[],[]], 'Fea_P4':[[],[]], 'Fea_P3':[[],[]], 'Fea_P2':[[],[]], 
    #  'Fea_P1':[[],[]], 'Fea_P5_Deconv':[[],[]], 'Fea_P4_Deconv':[[],[]], 'Fea_P3_Deconv':[[],[]], 'Fea_P2_Deconv':[[],[]],'Local_Fea':[[],[]], 
   #   'Local_Score':[[],[]], 'Global_Score':[[],[]]}#, 'conv2_dsn2':[[],[]], 'conv3_dsn2':[[],[]], 'conv4_dsn2':[[],[]], 'conv1_dsn1':[[],[]],
       #'conv2_dsn1':[[],[]], 'conv3_dsn1':[[],[]], 'score_dsn1_up':[[],[]], 'upscore_fuse':[[],[]]}
       #, 'h2_':[[],[]], 'h2_r':[[],[]], 'gate2':[[],[]],
       #'h3_':[[],[]], 'h3_r':[[],[]], 'gate3':[[],[]], 'h4_':[[],[]], 'h4_r':[[],[]], 'gate4':[[],[]], 'h5':[[],[]], 
       #'gate5':[[],[]], 'ch5_r':[[],[]], 'ch4_r':[[],[]], 'ch3_r':[[],[]], 'ch2_r':[[],[]], 'f1':[[],[]], 'f2':[[],[]],
        #'f3':[[],[]], 'f4':[[],[]], 'f5':[[],[]], 's5':[[],[]], 's4':[[],[]], 's3':[[],[]], 's2':[[],[]], 's1':[[],[]]}

      #'conv2_3':[[],[]],'conv2_3_':[[],[]],'conv2_4':[[],[]],'conv2_5':[[],[]],'conv2_5_':[[],[]],'gate2':[[],[]],
      #'decoder2_':[[],[]],'decoder1_':[[],[]],
      #'conv3_4_':[[],[]],
      # 'conv3_3_':[[],[]],'conv3_4':[[],[]],'conv2_3':[[],[]],'conv2_2_':[[],[]],'conv2_4':[[],[]],'conv2_5':[[],[]],'conv2_5_':[[],[]],'gate2':[[],[]],
      # 'conv3_5':[[],[]],'conv3_6':[[],[]],'conv3_6_':[[],[]],'gate3':[[],[]],'decoder2_':[[],[]],

for key in var_to_shape_map:
    #if key.find('step') > -1 :
     #   print ("tensor_name",key)

    str_name = key
    # 因为模型使用Adam算法优化的，在生成的ckpt中，有Adam后缀的tensor
    if str_name.find('Adam') > -1 or str_name.find('score') > -1 or str_name.find('decoder') > -1 or str_name.find('ga') > -1:
        continue
    #if str_name.find('fusion') > -1:
        #continue

    print('tensor_name:' , str_name)
    #new_name = 't_' + str_name
    #print('new_name:' , new_name)
    R[str_name]=reader.get_tensor(key)

    #if str_name.find('/') > -1:
    ######    names = str_name.split('/')
    #####    # first layer name and weight, bias
    ####    layer_name = names[0]
    ###    layer_add_info = names[1]
   ## else:
    #    layer_name = str_name
    ###    layer_add_info = None
##
   ## if layer_add_info == 'filter' or layer_add_info == 'W' or layer_add_info == 'weights':
  #      R[layer_name][0]=reader.get_tensor(key)
   # elif layer_add_info == 'b' or layer_add_info == 'biases':
    #    R[layer_name][1] = reader.get_tensor(key)
   # else:
    #    R[layer_name] = reader.get_tensor(key)

# save npy0 global_step
np.save('R.npy',R)
print('save npy over...')
#print(IAF['fc4'][0].shape)
#print(IAF['fc4'][1].shape)
