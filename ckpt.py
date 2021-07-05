import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
 
model_dir = "Model-123"
 
ckpt = tf.train.get_checkpoint_state(model_dir)
ckpt_path = ckpt.model_checkpoint_path
 
reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
param_dict = reader.get_variable_to_shape_map()
 
for key in param_dict.items(): #, val
    #try:
    #if key[0].find('ore') > -1:
        print(key[0])#, val
    #except:
        #pass
