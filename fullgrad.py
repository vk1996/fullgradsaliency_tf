import tensorflow as tf
from tensorflow.keras.models import load_model,Model
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from matplotlib import pyplot as plt 
from tensorflow.keras import backend as K
import numpy as np
import cv2


class Fullgrad():

    def __init__(self,base_model):
        ### instantiate base model ###
        self.base_model=base_model

    def create_fullgradmodel(self):
        ### create list with first element Input Layer since this layer has no bias ###
        self.layer_list=[self.base_model.layers[0].output]
        ### append layers having bias mostly conv layers with relu and final dense softmax ### 
        [self.layer_list.append(layer.output) for layer in self.base_model.layers if hasattr(layer,'bias')]
        ### some models last layer does not contain dense layer/bias as they have direct squashing non-linearity ###
        if not(isinstance(self.base_model.layers[-1],tf.keras.layers.Dense)):
          self.layer_list.append(self.base_model.layers[-1].output)
        self.full_grad_model=Model(self.base_model.input,self.layer_list)
 
        return self.full_grad_model

    def collect_initial_bias(self):
        ### foward pass of zero input flushes out feature map with bias alone ###
        self.input_shape=(1,self.base_model.layers[0].output.shape[1].value,self.base_model.layers[0].output.shape[2].value,self.base_model.layers[0].output.shape[3].value)
        return self.full_grad_model(np.zeros(shape=self.input_shape,dtype=np.float32))

    def postprocess(self,inputs):
        # Absolute value
        inputs = tf.math.abs(inputs)
        # Rescale operations to ensure gradients lie between 0 and 1
        inputs = inputs - tf.keras.backend.min(inputs)
        inputs = inputs / (tf.keras.backend.max(inputs)+K.epsilon())
        return inputs

    def postprocess_saliency(self,saliency_map):
        saliency_map = saliency_map - saliency_map.min()
        saliency_map = saliency_map / (saliency_map.max()+K.epsilon())
        saliency_map = saliency_map.clip(0,1)
        saliency_map=saliency_map.squeeze()
        #print(saliency_map.shape)
        saliency_map = np.uint8(saliency_map * 255)
        return saliency_map
    
    def saliency(self,preprocessed_input):
        
        init_bias=self.collect_initial_bias()
        
        ### fetch feature map with forward pass of pre-processed inputs ###
        features=self.full_grad_model(preprocessed_input)
        ### gather tensor with higher prob ###
        agg=tf.gather_nd(features[-1],[[0,tf.squeeze(tf.argmax(features[-1],axis=1))]])
        ### diff highest prob tensor w.r,t all input features
        grad=tf.gradients(agg,features)

        input_gradients=grad[0]
        bias_gradients=[]
        for i in range(1,len(grad)):
            bias_gradients.append(grad[i]*init_bias[i])

        ### multiply preprocessed input tensor with input gradient ###
        cam=tf.reduce_sum(self.postprocess(preprocessed_input*input_gradients),axis=-1)

        for i in range(1,len(grad)):
            grads=grad[i]
            ### check if input is feature map and not dense layers ###
            if len(grads.shape)>3:
                ### summation of features multiplied with their respective feature maps ### 
                cam=tf.math.add(cam,tf.reduce_sum(tf.image.resize(self.postprocess(grads*init_bias[i]),(self.input_shape[1],self.input_shape[2])),axis=-1))

        ''' the curse of TF graph refreshes all
        pretrained weights when global vars are init'''

        weights=self.full_grad_model.get_weights()
        sess=tf.Session()
        K.set_session(sess)
        sess.run(tf.global_variables_initializer ())
        self.full_grad_model.set_weights(weights)

        ### saliency feature map is retrieved from graph ###
        saliency_fmap=sess.run(cam)
        max_class=(np.argmax(sess.run(features[-1])))
        ### postprocess saliency to lie btw 0-255 ###
        return self.postprocess_saliency(saliency_fmap),max_class
        
        
