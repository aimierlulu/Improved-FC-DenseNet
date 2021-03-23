###############################################
# metrics and loss function
###############################################
import numpy as np
from keras import backend as K
# from sklearn.metrics import jaccard_similarity_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import jaccard_score, precision_score, recall_score, accuracy_score
import tensorflow as tf
import os
smooth_default = 0.


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

'''def precision(y_true, y_pred):  #我自己写的
    return precision_score(y_true, y_pred, average='binary')

def recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average='binary')
'''
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def dice_coef(y_true, y_pred, smooth=smooth_default, per_batch=False):
    if not per_batch:
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        #print(intersection.shape)
        #input()
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    else:
        y_true_f = K.batch_flatten(y_true)
        y_pred_f = K.batch_flatten(y_pred)
        intersec = 2. * K.sum(y_true_f * y_pred_f, axis=1, keepdims=True) + smooth
        union = K.sum(y_true_f, axis=1, keepdims=True) + K.sum(y_pred_f, axis=1, keepdims=True) + smooth
        return K.mean(intersec / union)


def jacc_coef(y_true, y_pred, smooth=smooth_default):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)


def jacc_loss(y_true, y_pred):
    return  1 - jacc_coef(y_true, y_pred)


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

#???????????????????????????????????
def dice_jacc_single(mask_true, mask_pred):
    bool_true = mask_true.reshape(-1).astype(np.bool)
    bool_pred = mask_pred.reshape(-1).astype(np.bool)
    if bool_true.shape != bool_pred.shape:
        raise ValueError("Masks of different sizes.")

    bool_sum = bool_true.sum() + bool_pred.sum()
    if bool_sum == 0:
        print("Empty mask")
        return 0.0
    intersec = np.logical_and(bool_true, bool_pred).sum()
    dice = 2. * intersec / bool_sum
    jacc = jaccard_score(bool_true.reshape((1, -1)), bool_pred.reshape((1, -1)),
                                    normalize=True, sample_weight=None)
 
    return dice, jacc

#我修正的
def dice_jacc_single1(mask_true, mask_pred):

    bool_true=mask_true.reshape(-1).astype(np.bool)
    bool_pred=mask_pred.reshape(-1).astype(np.bool)
    
    true_pos = np.logical_and(bool_true,bool_pred).sum()
    false_neg = np.logical_and(bool_true, (1-bool_pred)).sum()
    false_pos = np.logical_and((1-bool_true),bool_pred).sum()
    #print(true_pos)
    #print(false_neg)
    #print(false_pos)
    if bool_true.shape != bool_pred.shape:
        raise ValueError("Masks of different sizes.")

    bool_sum = bool_true.sum() + bool_pred.sum()
    if bool_sum == 0:
        print("Empty mask")
        return 0.0
  

    dice= 2* true_pos / (2*true_pos+ false_neg+false_pos)

    jacc = jaccard_score(bool_true.reshape((1, -1)), bool_pred.reshape((1, -1)),
                                    normalize=True, sample_weight=None)
 
    recall = true_pos/(true_pos+false_neg)
    
    precision = true_pos/(true_pos+false_pos+0.001)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return dice, jacc, recall, precision

def dice_jacc_mean(mask_true, mask_pred, smooth=0):
    dice = 0
    jacc = 0
    recall = 0
    precision = 0
    for i in range(mask_true.shape[0]):
        current_dice, current_jacc, current_recall, current_precision= dice_jacc_single1(mask_true=mask_true[i], mask_pred=mask_pred[i])
        dice = dice + current_dice
        jacc = jacc + current_jacc
        recall = recall + current_recall
        precision = precision + current_precision
    return dice/mask_true.shape[0], jacc/mask_true.shape[0], recall/mask_true.shape[0], precision/mask_true.shape[0]

'''def focal_loss(y_true, y_pred):
    pt_1=tf.where(tf.equal(y_true,1),y_pred,tf.ones_like(y_pred))
    pt_0=tf.where(tf.equal(y_true,0),y_pred,tf.zeros_like(y_pred))
    pt_1 = K.clip(pt_1, 1e-3, 0.999)
    pt_0 = K.clip(pt_0, 1e-3, 0.999)
    return -K.sum(0.988*K.pow(1.-pt_1,0)*K.log(pt_1))-K.sum((1-0.988)*K.pow(pt_0,0)*K.log(1.-pt_0))
'''
def tversky(y_true, y_pred):
    #y_pred:tensor
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    smooth=1e-3
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)
def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)
def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)


def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)
#版权声明：本文为CSDN博主「chestnut--」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
#原文链接：https://blog.csdn.net/m0_37477175/article/details/83004746
'''def focal_loss(y_true,y_pred):
    
    sess=tf.Session()
    #sess.run(tf.global_variables_initializer())

    print("118")
    print(y_pred)
    input()
    #sess = tf.Session()
    with sess.as_default():
      #sess.run(tf.global_variables_initializer())
      y_pred1=y_pred.eval() # tensor to numpy array
    #y_pred1=y_pred.numpy()
    print(y_pred1)
    input()
    erosion = cv2.erode(y_true,kernel,iterations = 1)
    dilation = cv2.dilate(y_true,kernel,iterations = 1)
    print(dilation)
    print(erosion)
    print(y_true)
    input()
    pt_1_weight=1.+(y_true-erosion)
    pt_0_weight=1.+(dilation-y_true)
    pt_1=tf.where(tf.equal(y_true,1),y_pred,tf.ones_like(y_pred))
    pt_0=tf.where(tf.equal(y_true,0),y_pred,tf.zeros_like(y_pred))
    pt_1 = K.clip(pt_1, 1e-3, 0.999)
    pt_0 = K.clip(pt_0, 1e-3, 0.999)
    
    return -K.sum(0.988*pt_1_weight*K.pow(1.-pt_1,0)*K.log(pt_1))-K.sum((1-0.988)*(pt_0_weight)*K.pow(pt_0,0)*K.log(1.-pt_0))
'''    
def focal_loss(gamma=0.,alpha=0.25):#gamma设置为0的时候就是加权，不设置为零可以处理难分样本,正类0.012,负类0.988
    def focal_loss_fixed(y_true,y_pred):
        pt_1=tf.where(tf.equal(y_true,1),y_pred,tf.ones_like(y_pred))
        pt_0=tf.where(tf.equal(y_true,0),y_pred,tf.zeros_like(y_pred))
        return -K.sum(alpha*K.pow(1.-pt_1,gamma)*K.log(pt_1))-K.sum((1-alpha)*K.pow(pt_0,gamma)*K.log(1.-pt_0))
    return focal_loss_fixed
