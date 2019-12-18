# fullgradsaliency_TF1.0
A Simple TF1.15 version of Full-Gradient Saliency Maps . The project investigates the concept 
of using input_gradients and bias_gradients to plot saliency maps of conv models.
This technique produces fine saliency maps when compared to ClassActivationMaps according 
to author , I will update with comparison in coming days.

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
#### tested asof now on vgg & resnet
from fullgrad import *
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input

K.clear_session()
model=ResNet50(weights='imagenet')

fullgrad=Fullgrad(model)
fullgrad_model=fullgrad.create_fullgradmodel()

cam,maxclass=fullgrad.saliency(preprocessed_input)

#### more detailed usage is available in example.ipynb ####

```
# TODO
```
Infer on various models and report any bugs.
The inference pipeline is bottlenecked by saving/loading 
weights due to tf.global_variable_init behaviour.Suggest 
someother idea to quicken the inference
```
