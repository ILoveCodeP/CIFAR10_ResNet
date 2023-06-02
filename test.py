#!/usr/bin/env python
# coding: utf-8

# In[1]:

from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.python.keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt
import tensorflow as tf

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torchvision import datasets, transforms
import torch
import os
import pickle
from scipy.stats import levy_stable

from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import load_model

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_filters', type=int)
parser.add_argument('--num_res', type=int)
args = parser.parse_args()
num_filter_block=args.num_filters
num_res = args.num_res

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
config = tf.compat.v1.ConfigProto()
config.allow_soft_placement=True
config.gpu_options.per_process_gpu_memory_fraction=0.7
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


# In[2]:

sigma = 0.05
noise_scale = np.sqrt(0.5)*sigma* 255

def stable_noise(img, alpha, beta, scale):
    '''
    此函数用将产生的stable噪声加到图片上
    传入:
        img   :  原图
        alpha  :  shape parameter
        beta :  symmetric parameter
        scale : scale parameter
        random_state : 随机数种子
    返回:
        stable_out : 噪声处理后的图片
        noise        : 对应的噪声
    '''
    # 产生stable noise
    noise = levy_stable.rvs(alpha=alpha, beta=beta, scale=scale, size=img.shape)
    # 将噪声和图片叠加
    stable_out = img + noise
    # 将超过 1 的置 1，低于 0 的置 0
    stable_out = np.clip(stable_out, 0, 255)
    # 取整
    stable_out = np.uint(stable_out)
    return stable_out, noise  # 这里也会返回噪声，注意返回值

def stable_noise_row(row, alpha, beta=0, scale=noise_scale):  # scale修改为0.03
    #         random_state = row.name #第0张图的随机数种子就是0，第1张图的随机数种子就是1，以此类推。。。
    return stable_noise(np.asarray(row).reshape(32, 32, 3), alpha, beta, scale)[0].reshape(3072)


def stable_noise_mixture(img, alphas, beta, scale):
    '''
    此函数用将产生的stable噪声加到图片上。mixture：对每个像素随机加不同alpha的噪声。
    传入:
        img   :  原图
        alpha  :  shape parameter
        beta :  symmetric parameter
        scale : scale parameter
        random_state : 随机数种子
    返回:
        stable_out : 噪声处理后的图片
        noise        : 对应的噪声
    '''
    noise = np.empty_like(img)
    # 产生stable noise
    for i in range(noise.shape[0]):
        for j in range(noise.shape[1]):
            alpha_c = np.random.choice(alphas)
            noise[i,j] = levy_stable.rvs(alpha=alpha_c,beta=beta,scale=scale)
    # 将噪声和图片叠加
    stable_out = img + noise
    # 将超过 255 的置 255，低于 0 的置 0
    stable_out = np.clip(stable_out, 0, 255)
    # 取整
    stable_out = np.uint(stable_out)
    return stable_out, noise # 这里也会返回噪声，注意返回值

def stable_noise_mixture_row(row, alphas, beta=0, scale=noise_scale):  #scale定为30
    '''
    对数据集中的行添加噪声
    row : pd.dataframe中的一行
    alpha, beta, scale : 需要添加噪声的alpha, beta, scale
    '''
    return stable_noise_mixture(np.asarray(row).reshape(32, 32, 3), alphas, beta, scale)[0].reshape(3072)
    # [0]是因为stable_noise这个函数会返回两个值，我们只需要第一个值

def stable_noise_hundred(inputs, alpha, beta=0, scale=noise_scale): 
    '''
    对数据集中的行添加噪声
    每一百行一起添加噪声
    alpha, beta, scale : 需要添加噪声的alpha, beta, scale
    '''
    noisy_data = np.zeros(inputs.shape)
    num_samples = np.shape(inputs)[0]
    k = np.arange(0,num_samples,100)
    for i in k:
        temp = inputs[i:i+100]
        temp_noise = stable_noise(temp, alpha, beta, scale)[0]
        noisy_data[i:i+100] = temp_noise
    return noisy_data

def stable_noise_mixture_hundred(inputs, alphas, beta, scale):

    noisy_data = np.zeros(inputs.shape)
    num_samples = np.shape(inputs)[0]
    k = np.arange(0,num_samples,100)
    for i in k:
        temp = inputs[i:i+100]
        alpha_c = np.random.choice(alphas, size=temp.shape)

        # 产生stable noise
        noise = levy_stable.rvs(alpha=alpha_c,beta=beta,scale=scale)
        # 将噪声和图片叠加
        stable_out = temp + noise
        # 将超过 255 的置 255，低于 0 的置 0
        stable_out = np.clip(stable_out, 0, 255)
        # 取整
        stable_out = np.uint(stable_out)
        noisy_data[i:i+100] = stable_out
    return noisy_data


# In[4]:


# 建立基于keras的cnn模型
# Conv2D(64, 8×8) – Conv2D(128, 6×6) – Conv2D(128, 5×5) – Softmax(10)


def resnet_block(inputs, num_filters=num_filter_block,
                    kernel_size=3, strides=1,
                    activation='relu'):
    x = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same',
                kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    if (activation):
        x = Activation('relu')(x)
    return x


# 建一个20层的ResNet网络
def resnet_v1(input_shape,num_res):
    inputs = Input(shape=input_shape)  # Input层，用来当做占位使用

    # 第一层
    x = resnet_block(inputs)
    print('layer1,xshape:', x.shape)
    # 第2~7层
    for i in range(num_res):
        a = resnet_block(inputs=x)
        b = resnet_block(inputs=a, activation=None)
        x = keras.layers.add([x, b])
        x = Activation('relu')(x)
    # out：32*32*16
    # 第8~13层
    for i in range(num_res):
        if i == 0:
            a = resnet_block(inputs=x, strides=2, num_filters=2*num_filter_block)
        else:
            a = resnet_block(inputs=x, num_filters=2*num_filter_block)
        b = resnet_block(inputs=a, activation=None, num_filters=2*num_filter_block)
        if i == 0:
            x = Conv2D(2*num_filter_block, kernel_size=3, strides=2, padding='same',
                        kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
        x = keras.layers.add([x, b])
        x = Activation('relu')(x)
    # out:16*16*32
    # 第14~19层
    for i in range(num_res):
        if i == 0:
            a = resnet_block(inputs=x, strides=2, num_filters=4*num_filter_block)
        else:
            a = resnet_block(inputs=x, num_filters=4*num_filter_block)

        b = resnet_block(inputs=a, activation=None, num_filters=4*num_filter_block)
        if i == 0:
            x = Conv2D(4*num_filter_block, kernel_size=3, strides=2, padding='same',
                        kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
        x = keras.layers.add([x, b])  # 相加操作，要求x、b shape完全一致
        x = Activation('relu')(x)
    # out:8*8*64
    # 第20层
    x = AveragePooling2D(pool_size=2)(x)
    # out:4*4*64
    y = Flatten()(x)
    # out:1024
    outputs = Dense(10, activation='softmax',
                    kernel_initializer='he_normal')(y)

    # 初始化模型
    # 之前的操作只是将多个神经网络层进行了相连，通过下面这一句的初始化操作，才算真正完成了一个模型的结构初始化
    model = Model(inputs=inputs, outputs=outputs)
    return model


# In[12]:


# In[ ]:


train_list=['mix',6,0,2,1.9,1.5,1.3,1,0.9,0.5,5.5,'mixwithout0.5']
alpha_trains = [2,1.9,1.5,1.3,1,0.9,0.5]
alpha_trains2 = [2,1.9,1.5,1.3,1,0.9]
num = len(alpha_trains)
times_tmp = 2 #multiple时 每种噪声加几倍
model_type = 'resnet'

for alpha_train in train_list:
    acc_temp = []
    auc_temp = []
    mi_temp = []
    alpha_test_temp = []
    num_res = num_res #原始值是6
    for repeat_time in  range(5): 
        model = load_model('resnet_trainalpha{}_repeattimes{}_numfilters{}_numblocks{}.h5'.format(alpha_train,repeat_time,num_filter_block,num_res))
        noisy_test = np.tile(X_test_f, (times,1))
        (X_train_temp, y_train), (X_test_temp, y_test_f) = cifar10.load_data()
        num_trainSamples = X_train_temp.shape[0]
        num_testSamples = X_test_temp.shape[0]
        times = 10
        X_train = X_train_temp.reshape(num_trainSamples, -1)
        X_test_f = X_test_temp.reshape(num_testSamples, -1)
        noisy = np.tile(X_train, (times,1))
        print(np.shape(noisy))
        y_tr = np.tile(y_train,(times+1,1))
        num_classes = 10
        encoder = LabelEncoder()
        nb_iterations = 120000
        batch_size = 32

        gc.collect()

        alpha_tests = ['mix',6,0,2,1.9,1.5,1.3,1,0.9,0.5]
        # alpha_tests=[6,0,2,1.9]
        for alpha_test in alpha_tests:
            print(alpha_test)
            X_test = X_test_f.copy()
            y_test = y_test_f.copy()
            if alpha_test == 0:
                ########ONLY REPEAT ONCE FOR CLEAN DATA TEST###########
                y_te = y_test.copy()
                y_test = encoder.fit_transform(y_te)
                y_test = to_categorical(y_test, num_classes)
                y_test = np.array(y_test)
                y_test = y_test.reshape(10000,10)
                ########################################################
                # X_test = np.r_[X_test,noisy_test]
                # X_test = X_test.reshape(10000*(times+1),32,32,3)
                X_test = X_test.reshape(10000,32,32,3)
            elif alpha_test == 6:
                noisy_te = np.tile(X_test, (times_tmp,1))#50000*times_tmp,3072
                X_test = X_test.reshape(num_testSamples,32,32,3)
                for alpha_ in alpha_trains:
                    temp = noisy_te.copy() #50000*times_tmp*3072
                    noise = stable_noise_hundred(temp,alpha=alpha_,beta=0,scale = noise_scale)
                    noise = noise.reshape(num_testSamples*times_tmp,32,32,3)
                    X_test = np.r_[X_test,noise]
                    print(np.shape(X_test))
                y_te = np.tile(y_test, (times_tmp*num+1,1))
                y_test = encoder.fit_transform(y_te)
                y_test = to_categorical(y_test, num_classes)
                y_test = np.array(y_test)
                y_test = y_test.reshape(num_testSamples*(times_tmp*num+1),10)

            elif alpha_test == 'mix':
                y_te = np.tile(y_test,(times+1,1))
                y_test = encoder.fit_transform(y_te)
                y_test = to_categorical(y_test, num_classes)
                y_test = np.array(y_test)
                y_test = y_test.reshape(10000*(times+1),10)

                X_noise_test = stable_noise_mixture_hundred(noisy_test,alphas = alpha_trains,beta=0,scale = noise_scale)
                X_noise_test = X_noise_test.reshape(10000*times,32,32,3)
                X_test = X_test.reshape(10000,32,32,3)
                X_test = np.r_[X_test,X_noise_test]
            else:
                y_te = np.tile(y_test,(times+1,1))
                y_test = encoder.fit_transform(y_te)
                y_test = to_categorical(y_test, num_classes)
                y_test = np.array(y_test)
                y_test = y_test.reshape(10000*(times+1),10)
                X_noise_test = stable_noise_hundred(noisy_test,alpha=alpha_,beta=0,scale = noise_scale)
                # X_noise_test = np.apply_along_axis(lambda x: stable_noise_row(x, alpha=alpha_test, scale=noise_scale),axis=1, arr=noisy_test)
                X_noise_test = X_noise_test.reshape(10000*times,32,32,3)
                X_test = X_test.reshape(10000,32,32,3)
                X_test = np.r_[X_test,X_noise_test]
            
            X_test = X_test/ 255
            for j in range(3):
                X_test[:, :, :, j] = (X_test[:, :, :, j] - np.mean(X_test[:, :, :, j])) / np.std(X_test[:, :, :, j])

                    
            y_pred = model.predict(X_test)
            tt1=np.argmax(y_test, axis=1)
            auc_value = metrics.roc_auc_score(y_test,y_pred,multi_class='ovo',average='macro')


            tt2=np.argmax(y_pred, axis=1)
            acc = metrics.accuracy_score(tt1, tt2)



            acc_temp.append(acc)
            auc_temp.append(auc_value)
            alpha_test_temp.append(alpha_test)
            del X_test
            gc.collect()

        del model
        gc.collect()
    
    with open("./results/CIFAR10_resnet_numfilters{}_b{}_alpha{}_sigma{}_num_blocks{}.txt".format(args.num_filter_block,batch_size,alpha_train,sigma,num_res), "wb") as path_acc:
        pickle.dump(acc_temp, path_acc)
    # np.savetxt('accuracy_alpha{}_model{}_numfilter{}.txt'.format(alpha_train,model_type,num_filter_block), acc_temp)
    # np.savetxt('auc_alpha{}_model{}_numfilter{}.txt'.format(alpha_train,model_type,num_filter_block), auc_temp)
    # np.savetxt('alpha_test__alpha{}_model{}_numfilter{}.txt'.format(alpha_train,model_type,num_filter_block), alpha_test_temp)



