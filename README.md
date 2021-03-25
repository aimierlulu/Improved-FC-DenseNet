# Improved-FC-DenseNet
A new fully automatic coronary artery segmentation method based on an improved FC-DenseNet is presented. The proposed method is efficiently trained end-to-end, using CCTA images and ground truth masks to make a per-pixel semantic inference. 






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
