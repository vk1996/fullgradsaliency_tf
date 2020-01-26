# fullgradsaliency_TF1.0
A Simple TF1.15 version of Full-Gradient Saliency Maps . The project investigates the concept 
of using input_gradients and bias_gradients to plot saliency maps of conv models.
This technique produces fine saliency maps when compared to ClassActivationMaps according 
to author , I will update with comparison in coming days.

![alt text](https://github.com/vk1996/fullgradsaliency_TF1.0/blob/master/pngs/download.png)
![alt text](https://github.com/vk1996/fullgradsaliency_TF1.0/blob/master/pngs/download%20(2).png)


# source 
```
https://github.com/idiap/fullgrad-saliency
@inproceedings{srinivas2019fullgrad,
    title={Full-Gradient Representation for Neural Network Visualization},
    author={Srinivas, Suraj and Fleuret, François},
    booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
    year={2019}
}

````
# Usage

```
'''
Tested on vgg , resnet, densenet, Xception
Feel free to use in custom models and other 
architectures and report issues
'''

from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras import backend as K
import numpy as np
import os
os.chdir('fullgradsaliency_TF1.0')
from fullgrad import FullGrad

K.clear_session()
base_model=ResNet50(weights='imagenet')

fullgrad=FullGrad(base_model)


input_=np.ones(shape=(1,224,224,3))
preprocessed_input=preprocess_input(input_)

'''
check if completeness test is satisfied. Refer example.ipynb 
'''
fullgrad.checkCompleteness(input_)

'''
now get saliency map of highest class from fullgrad model
'''

saliency=fullgrad.saliency(preprocessed_input)
saliency=fullgrad.postprocess_saliency_map(saliency[0])

'''
more detailed usage is available in example.ipynb
'''

```
# TODO
```
Infer on various models and report any bugs.
Compare the results of model by editing or
removing the part of image having high 
confidence on saliency heatmap.
```
