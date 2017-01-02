# Xvision

This repo contains the implementation of **Transfer Learning** for Chest Xray image analysis using **Deep learning** with Tensorflow.

The **maxpool-5** layer of a pretrained **VGGNet-16** model has been used as the feature extractor here and then further trained on a 2-layer Deep neural network for classification of **Normal vs Nodular** Chest Xray Images.

## Nodular vs Normal Chest Xray
![Alt text](https://github.com/ayush1997/Xvision/blob/master/node.jpg "Optional Title")
![Alt text](https://github.com/ayush1997/Xvision/blob/master/normal.jpg "Optional Title")

## Some specifications

| Property      |Values         |
| ------------- | ------------- |
| Pretrained Model | VggNet-16  |
| Optimizer used  | stochastic gradient descent(SGD)  |
| Learning rate  | 0.01|  
|Mini Batch Size| 20 |
| Epochs | 20 |





