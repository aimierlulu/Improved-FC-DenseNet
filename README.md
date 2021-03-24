# Improved-FC-DenseNet
A new fully automatic coronary artery segmentation method based on an improved FC-DenseNet is presented. The proposed method is efficiently trained end-to-end, using CCTA images and ground truth masks to make a per-pixel semantic inference. 


Model:


The provided model is basically a FC-DenseNet,we present an improved FC-DenseNet for fully automatic coronary artery segmentation. In the downsampling path, we have added a convolutional layer with kernel size of 5  5 to realize a more accurate location. Additionally, direct down skip connections have been added to assure maximum information flow between layers. In the upsampling path, we have improved the TU blocks, aiming at extracting richer semantic features.  .



Training:


The segmentation method was implemented in Python on a computer equipped with two NVIDIA 1080ti graphics cards, each of which has 12GB memory. The Keras library served as a high-level framework, running over TensorFlow. It was trained for 100 epochs, where each epoch took 3600 seconds. 


Hyper-parameter	Value:
+ Epoch ---->100
+ Learning rate	---->0.00006
+ Batch Normalization	---->True
+ Batch size	---->1
+ Loss function	---->Tversky loss
+ Optimization algorithm ---->Adam
+ Up-sampling	---->Deconv


Code:
+ train: python train.py
+ test: python test.py
+ predict: python predict.py
