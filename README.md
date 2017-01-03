# Xvision

Chest Xray image analysis using **Deep Learning** and  exploiting **Deep Transfer Learning** technique for it with Tensorflow.

The **maxpool-5** layer of a pretrained **VGGNet-16(Deep Convolutional Neural Network)** model has been used as the feature extractor here and then further trained on a **2-layer Deep neural network** for classification of **Normal vs Nodular** Chest Xray Images.

## Nodular vs Normal Chest Xray
<img src="https://github.com/ayush1997/Xvision/blob/master/image/node.jpg" width="300" height="300" />
<img src="https://github.com/ayush1997/Xvision/blob/master/image/normal.jpg" width="300" height="300" />

## Some specifications

| Property      |Values         |
| ------------- | ------------- |
| Pretrained Model | VggNet-16  |
| Optimizer used  | stochastic gradient descent(SGD)  |
| Learning rate  | 0.01|  
|Mini Batch Size| 20 |
| Epochs | 20 |
|GPU trained on| Nvidia GEFORCE 920M|

## Evaluation
### Confusion Matrix and Training Error Graph

<img src="https://github.com/ayush1997/Xvision/blob/master/image/cfm.jpg" width="450" height="400" />
<img src="https://github.com/ayush1997/Xvision/blob/master/image/nodule.jpg" width="400" height="400" />

|     |  **Normal** | **Nodule** |
|------|---------|---------|
| **Precision**| 0.7755102| 0.55555556 |
|**Recall**| 0.76 | 0.57692308 |

**Accuracy** 69.3333 %

## DataSet
[openi.nlm.nih.gov](https://openi.nlm.nih.gov/gridquery.php?q=&it=x,xg&sub=x&m=1&n=101) has a large base of Xray,MRI, CT scan images publically available.Specifically Chest Xray Images have been scraped, Normal and Nodule labbeled images are futher extrated for this task.

## How to use ?
The above code can be used for **Deep Transfer Learning** on any Image dataset to train using VggNet as the PreTrained network. 
### Steps to follow 

1. Download Data- the script download images and saves corresponfing disease label in json format.

  ```python scraper.py <path/to/folder/to/save/images>```

2. Follow the ```scraper/process.ipynb``` notebook for Data processing and generate

  * Training images folder - All images for training
  * Testing images Folder - All images for testing
  * Training image labels file - Pickled file with training labels
  * Testing image labels file - Pickled file with testing labels

3. Extract features(**CNN Codes**) from the **maxpool:5** layer of PreTrained CovNet(VggNet) and save them beforehand for faster training of Neural network.

    ```python train.py <Training images folder> <Testing image folder> <Train images codes folder > <Test images codes folder>```

4.  The extracted features are now used for training our **2-Layer Neural Network** from scratch.The computed models are saved as tensorflow checkpoint after every **Epoch**.

    ```python train_model.py <Training images folder> <Train images codes folder> <Training image labels file> <Folder to         save models>```

5.  Finally the saved models are used for making predictions.Confusion Matrix is used as the Performance Metrics for this classifcation task.

    ```python test_model.py <Testing images folder> <Test images codes folder> <Testing image labels file> <Folder with saved
    models>```

## Some Predictions

![Alt text](https://github.com/ayush1997/Xvision/blob/master/image/pred.jpg "Optional Title")

## References

> [Learning to Read Chest X-Rays: Recurrent Neural Cascade Model for Automated Image Annotation](https://arxiv.org/pdf/1603.08486.pdf)

> [Deep Convolutional Neural Networks for Computer-Aided Detection: CNN Architectures,
Dataset Characteristics and Transfer Learning](https://arxiv.org/pdf/1602.03409.pdf)

## Contribute

If you want to contribute and add new feature feel free to send Pull request [here](https://github.com/ayush1997/Xvision/pulls)
To report any bugs or request new features, head over to the [Issues page](https://github.com/ayush1997/Xvision/issues)
