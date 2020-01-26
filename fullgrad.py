'''
Copyright 2020 Vignesh Kotteeswaran <iamvk888@gmail.com>
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

''' Tensorflow version of FullGrad saliency algorithm with gradient completeness check and single
    skeleton to create FullGrad model of different backbones'''


import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import numpy as np
import cv2


class FullGrad():
  def __init__(self,base_model,num_classes=1000,class_names=None,verbose=False):
    self.base_model=base_model
    self.num_classes=num_classes
    self.model=self.linear_output_model(self.base_model)
    self.verbose=verbose
    assert(self.num_classes>0),'Output classes must be greater than 1 but found'+str(self.num_classes)
    self.blockwise_biases = self.getBiases()
    self.check=True
    self.class_names=class_names
    if self.class_names != None:
      assert(len(self.class_names)==self.num_classes),'Num classes and class names not matched'
    else:
      self.class_names= [None]*self.num_classes

  def linear_output_model(self,model):
    x=Dense(self.num_classes)(model.layers[-2].output)
    temp=Model(model.input,x)
    temp.set_weights(self.base_model.get_weights())
    return temp
  
  def getBiases(self):
    reset_act=False
    self.biases = []
    for count,layer in enumerate(self.model.layers):
      if count==0:
        self.biases.append(0)
        x=np.zeros(shape=(1,self.model.layers[0].input_shape[0][1],self.model.layers[0].input_shape[0][2],self.model.layers[0].input_shape[0][3])).astype(np.float32)
      if isinstance(layer,Conv2D) or isinstance(layer,Dense):
        #print(layer.get_config()['activation'])
        if layer.get_config()['activation'] != 'linear':
          reset_act=True
          #old_act=layer.get_config()['activation']
          old_act='relu'
          layer.activation=Activation('linear')
        if len(layer.input_shape)==2:
          x=layer(np.zeros(shape=(1,layer.input_shape[1])).astype(np.float32))
          self.biases.append(x)
        if len(layer.input_shape)==4:
          x=layer(np.zeros(shape=(1,layer.input_shape[1],layer.input_shape[2],layer.input_shape[3])).astype(np.float32))
          self.biases.append(x)
        if reset_act==True:
          layer.activation=Activation(old_act)
          reset_act=False
      if isinstance(layer,BatchNormalization):
        x=layer(np.zeros(shape=(1,layer.input_shape[1],layer.input_shape[2],layer.input_shape[3])).astype(np.float32))
        self.biases.append(x)

    return self.biases
  
  def getFeatures(self,input):
    self.features=[]
  
    values=[i for i in np.zeros(shape=(len(self.model.layers))).astype(np.float32)]
    keys=[str(i.output.name) for i in self.model.layers]
    data=dict(zip(keys,values))
    #print(data)

    for count,layer in enumerate(self.model.layers):
      if count==0:
        if not isinstance(input,tf.Tensor):
          #input=tf.convert_to_tensor(input_)
          data[layer.output.name]=tf.convert_to_tensor(input)
        self.features.append(input)
      if count==1:
        #print(layer)
        
        if isinstance(layer,Dense) or isinstance(layer,Conv2D):
        
          #print('dense conv')
          if layer.get_config()['activation'] != 'linear':
            #old_act=layer.get_config()['activation']
            old_act='relu'
            #print(old_act)
            layer.activation=Activation('linear')
            x=layer(input)
            self.features.append(x)
            data[layer.output.name]=Activation(old_act,name='cut_'+old_act+'_conv'+str(count))(x)

          if layer.get_config()['activation'] == 'linear':
            #print(layer.name,layer,layer.input.name,layer.output.name)
            data[layer.output.name]=layer(input)
            self.features.append(data[layer.output.name])
                 
        elif isinstance(layer,BatchNormalization):
          #print('BN')
          #data[layer.output.name]=layer(input)
          data[layer.output.name]=layer(data[layer.input.name])
          self.features.append(data[layer.output.name])

        else:
          data[layer.output.name]=layer(input)

      if count>1:
        if isinstance(layer,Add) or isinstance(layer,Concatenate):
          data[layer.output.name]=layer([data[i.name] for i in layer.input ])
            
        elif isinstance(layer,Conv2D):
          if layer.get_config()['activation'] != 'linear':
            #old_act=layer.get_config()['activation']
            old_act='relu'
            #print(old_act)
            layer.activation=Activation('linear')
            x=layer(data[layer.input.name])
            self.features.append(x)
            data[layer.output.name]=Activation(old_act,name='cut_'+old_act+'_conv'+str(count))(x)
        
          else:
            data[layer.output.name]=layer(data[layer.input.name])
            self.features.append(data[layer.output.name])

        
        #data[layer.name]=layer(data[layer.input.name.split('/')[0]])
        elif isinstance(layer,Dense):
          if layer.get_config()['activation'] != 'linear':
            #old_act=layer.get_config()['activation']
            old_act='relu'
            #print(old_act)
            layer.activation=Activation('linear')
            x=layer(data[layer.input.name])
            self.features.append(x)
            data[layer.output.name]=Activation(old_act,name='cut_'+old_act+'_dense'+str(count))(x)
        
          else:
            data[layer.output.name]=layer(data[layer.input.name])
            self.features.append(data[layer.output.name])

        #data[layer.name]=layer(data[layer.input.name.split('/')[0]])
        
        elif isinstance(layer,BatchNormalization):
          #print(layer.name,layer,layer.input.name,layer.output.name)
          data[layer.output.name]=layer(data[layer.input.name])
          self.features.append(data[layer.output.name])
        
        else:
        #print(layer)
          data[layer.output.name]=layer(data[layer.input.name])
      
    lastname=layer.output.name
    #return data[lastname],self.features
    self.layer_data=data 
    #lastname=layer.name
    return data[lastname],self.features

  def fullGradientDecompose(self, image, target_class=None):
        """
        Compute full-gradient decomposition for an image
        """
        out, features = self.getFeatures(image)
        ### out--> imagenet 1000 probs , features--> conv_block_features ####

        if target_class is None:
            target_class = tf.argmax(out,axis=1)
      
        if self.check:
          check_target_class=K.eval(target_class)[0]
          print('class:',check_target_class,'class name:',self.class_names[check_target_class])
          self.check=False
        assert(len(features)==len(self.blockwise_biases)),'Number of features {} not equal to number of blockwise biases {}'.format(len(features),len(self.blockwise_biases))
        agg=tf.gather_nd(features[-1],[[0,tf.squeeze(tf.argmax(features[-1],axis=1))]])
        gradients=tf.gradients(agg,features)
        #print('gradients:',gradients)
        
        for grad in gradients:
          if grad==None and self.verbose:
            print(grad)
          if grad!=None and self.verbose:
            print('grad:',grad.shape)
        
        #print(gradients[0])
        # First element in the feature list is the image
        input_gradient = gradients[0]*image
 
        # Loop through remaining gradients
        bias_gradient = []
        for i in range(1, len(gradients)):
            #print('mul:',gradients[i].shape,)
            #print('mul:',input_grad[0].shape,image.shape,'max:',input_grad[0].max(),image.max())
            if self.verbose:
              print('mul:',gradients[i].shape,self.blockwise_biases[i].shape,'max:',K.eval(gradients[i]).max(),self.blockwise_biases[i].max())
            bias_gradient.append(gradients[i] * self.blockwise_biases[i]) 
        
        return (input_gradient),bias_gradient,out

  def checkCompleteness(self,input):
        self.check=True
        #print('starting completeness test')
        #Random input image
        #input=np.random.randn(1,224,224,3).astype(np.float32)
        input=tf.convert_to_tensor(input.astype(np.float32))
        
        # Compute full-gradients and add them up
        
        input_grad, bias_grad,raw_output = self.fullGradientDecompose(input, target_class=None)
        
        #input_grad=K.eval(input_grad)
        if self.verbose:
          print(input_grad.sum(),input_grad.max())
          #print(K.eval(i).sum() for i in bias_grad)
        #print(input_grad,input)
        
        fullgradient_sum = tf.reduce_sum(input_grad)
        for i in range(len(bias_grad)):
            if self.verbose:
              print('fullgrad sum:',K.eval(fullgradient_sum),'biasgrad sum:',K.eval(bias_grad[i]).sum(),K.eval(bias_grad[i]).max(),K.eval(bias_grad[i]).shape)
            fullgradient_sum += tf.reduce_sum(bias_grad[i])

        raw_output=K.eval(raw_output)
        fullgradient_sum=K.eval(fullgradient_sum)
        
        print('Running completeness test.....')
        print('final_layer_max_class_linear_output:',raw_output.max())
        print('sum of FullGrad:', fullgradient_sum)

        # Compare raw output and full gradient sum
        err_message = "\nThis is due to incorrect computation of bias-gradients.Saliency may not completely represent input&bias gradients, use at your own risk "
        err_string = "Completeness test failed! Raw output = " + str(raw_output.max()) + " Full-gradient sum = " + str(fullgradient_sum)  
        assert np.isclose(raw_output.max(), fullgradient_sum,atol=0.00001), err_string + err_message
        print('Completeness test passed for FullGrad.')  

  def postprocess(self,inputs):
        # Absolute value
        inputs = tf.math.abs(inputs)
        # Rescale operations to ensure gradients lie between 0 and 1
        inputs = inputs - tf.keras.backend.min(inputs)
        inputs = inputs / (tf.keras.backend.max(inputs)+K.epsilon())
        return inputs

  
  def saliency(self, image, target_class=None):
        #FullGrad saliency
        input_grad, bias_grad,_ = self.fullGradientDecompose(tf.convert_to_tensor(image.astype(np.float32)), target_class=target_class)

        
        # Input-gradient * image
        #print('input_mul:',input_grad[0].shape,image.shape,'max:',input_grad.max(),image.max())
        grd = input_grad * tf.convert_to_tensor(image.astype(np.float32))
        gradient = tf.reduce_sum(self.postprocess(grd),axis=-1)
        ### input grad postprocessed and summed across axis 1 ###
        cam = gradient
        #print(cam.shape)
        
        # Bias-gradients of conv layers
        for i in range(len(bias_grad)):
            # Checking if bias-gradients are 4d / 3d tensors
            if len(bias_grad[i].shape) == len(image.shape): 
                #temp = self.postprocess(bias_grad[i])
                if self.verbose:
                  print(temp.shape,image.shape)
                if len(image.shape) ==4:
                    #gradient=skimage.transform.resize(temp,(temp.shape[0],image.shape[1],image.shape[2],temp.shape[-1]))
                    cam=tf.math.add(cam,tf.reduce_sum(tf.image.resize(self.postprocess(bias_grad[i]),(self.model.layers[0].input_shape[0][1],self.model.layers[0].input_shape[0][2])),axis=-1))
                #print(cam.shape,gradient.shape,gradient.sum(1, keepdim=True).shape)
                #cam += tf.reduce_sum(gradient,axis=-1)
                if self.verbose:
                  print(cam.shape,gradient.shape)

        return K.eval(cam)

  def postprocess_saliency_map(self,saliency_map):
    saliency_map = saliency_map
    saliency_map = saliency_map - saliency_map.min()
    saliency_map = saliency_map / saliency_map.max()
    saliency_map = saliency_map.clip(0,1)
    saliency_map=saliency_map.squeeze()
    #print(saliency_map.shape)
    saliency_map = np.uint8(saliency_map * 255)
    return saliency_map
