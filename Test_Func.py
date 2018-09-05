# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 12:17:48 2018

@author: s161488
"""
import tensorflow as tf
import numpy as np
import os
from cifar10 import Read_Data_from_Record 
from resnet_simple import mix_up_process, build_tower
from resnet_simple_basic import build_tower as build_tower_basic
LABLE_SPACE = ["airplane", "automobile", "bird", "cat", "deer","dog","frog","horse","ship","trunk"]
#mix_up_process(x_input, target_input, lambda_input, mixup_option)
def Test_Input():
    data_dir = ["/zhome/1c/2/114196/Documents/SA/tfrecord_data/train.tfrecords"]
    batch_size = 5
    augmentation = False
    with tf.Graph().as_default():
        image_batch, label_batch = Read_Data_from_Record(data_dir, batch_size, augmentation, shuffle = False)
        init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
        with tf.Session() as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord = coord)
            print("start fetching data")
            image_read, label_read = sess.run([image_batch, label_batch])
            coord.request_stop()
            coord.join(threads)
        
    return image_read, label_read
    
def Plot_Image(image_input, label_input):
    plt.figure(0)
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.imshow(image_input[i].astype('uint8'))
        plt.title(LABLE_SPACE[label_input[i]])
    plt.show()
    
def Test_Mixup_Func():
    x_input = np.random.randint(0,10,[5,3,3,3])
    target_input = np.random.randint(10,15,[5])
    lambda_input = 0.5
    mixup_option = True
    x_input_tf = tf.constant(x_input, dtype = tf.float32)
    target_input_tf = tf.constant(target_input, dtype = tf.int64)
    result_output, indices = mix_up_process(x_input_tf, target_input_tf, lambda_input, mixup_option)
    x_out_tf, y_out_tf = result_output
    with tf.Session() as sess:
        x_out, y_out = sess.run([x_out_tf, y_out_tf])
    #-----------
    y_out_num = target_input[indices]
    x_out_num = x_input*lambda_input + (1-lambda_input)*x_input[indices]
    #--------
    print(np.sum(np.reshape(y_out_num-y_out,[-1])))
    print(np.sum(np.reshape(x_out_num-x_out,[-1])))
    print(y_out,y_out_num)
    print(x_out[0],x_out_num[0])
    
def Test_ResNet_Func():
    #build_tower(images, labels, input_lambda, NUM_CLASS, is_training, Selec_Layer_Index)
    images = tf.constant(0.1, shape = [5,32,32,3], dtype = tf.float32)
    labels = tf.constant(1, shape = [5], dtype = tf.int64)
    NUM_CLASS = 10
    is_training = tf.placeholder(tf.bool)
    Selec_Layer_Index = tf.placeholder(tf.int64)
    input_lambda = tf.placeholder(tf.float32, shape = [1])
    logits, target = build_tower(images, labels, input_lambda, NUM_CLASS, is_training, Selec_Layer_Index)
    print("----------------------logits:",logits)
    print("----------------------target:",target)
    
def Check_Mixup_Random_Indices_In_ResNet():
    x_input = np.random.random([5,32,32,3])
    target_input = np.random.randint(10,15,[5])
    
    images = tf.placeholder(shape = [5,32,32,3], dtype = tf.float32)
    labels = tf.placeholder(shape = [5], dtype = tf.int64)
    NUM_CLASS = 10
    is_training = tf.placeholder(tf.bool)
    Selec_Layer_Index = tf.placeholder(tf.int64)
    input_lambda = tf.placeholder(tf.float32, shape = [1])
    logits, target, ind1, ind2, ind3 = build_tower(images, labels, input_lambda, NUM_CLASS, is_training, Selec_Layer_Index)
    print(logits, target, ind1, ind2, ind3)
    with tf.Session() as sess:
        feed_dict = {images:x_input,
                     labels:target_input,
                     is_training: True,
                     Selec_Layer_Index:3,
                     input_lambda:[0.5]}
        targ_num, ind_1_num, ind_2_num, ind_3_num = sess.run([target, ind1, ind2, ind3], feed_dict = feed_dict)
    print(ind_1_num, ind_2_num, ind_3_num)
    print(targ_num, target_input[ind_2_num])
    
def Collect_Statistics():
    path_mom = "/scratch/mixup"
    path_all = os.listdir(path_mom)
    path_selec = [v for v in path_all if 'Iter_' in v and '_preactivation' not in v]
    path_selec = np.sort(path_selec)
    train_stat_tot = []
    val_stat_tot = []
    print(path_selec)
    for single_path in path_selec:
        print(single_path)
        path_son = os.path.join(path_mom, single_path)
        train_stat = np.load(os.path.join(path_son, 'trainstat.npy'))
        val_stat = np.load(os.path.join(path_son,'valstat.npy'))
        train_stat_tot.append(train_stat)
        val_stat_tot.append(val_stat)
    np.save(os.path.join(path_mom, 'train_stat'), train_stat_tot)
    np.save(os.path.join(path_mom, 'val_stat'), val_stat_tot)
    
def Check_Stat():
    import matplotlib.pyplot as plt
    path_mom = "/home/s161488/manifold_mixup/stat"
    train_stat = np.load(os.path.join(path_mom, 'train_stat.npy'))
    val_stat = np.load(os.path.join(path_mom, 'val_stat.npy'))
    
    train_stat_mean = []
    for single_train in train_stat:
        length = np.shape(single_train)[0]
        index = np.multiply(range(length//300),300)
        train_stat_mean.append(Calc_Mean(index, single_train))
    val_stat_mean = []
    for single_val in val_stat:
        row = np.min(np.where(np.mean(single_val, axis = 1)==0))
        index = np.multiply(range(row//2),2)
        val_stat_mean.append(Calc_Mean(index, single_val[:row,:]))
        
    fig = plt.figure(figsize = (11,4))
    ax_global = fig.add_subplot(111)
    ax_global.spines['top'].set_color('none')
    ax_global.spines['bottom'].set_color('none')
    ax_global.spines['left'].set_color('none')
    ax_global.spines['right'].set_color('none')
    ax_global.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

    color_space = ['r','g','b']
    legend_space = ['No Mixup', 'Input Mixup', 'Manifold Mixup']
    ax = fig.add_subplot(1,2,1)
    for i in range(3):
        x_index = np.multiply(range(np.shape(val_stat_mean[i])[0]),300)
        ax.semilogy(x_index, val_stat_mean[i][:,0], color = color_space[i], ls = '-', lw = 2)
    ax.ticklabel_format(style = 'sci', axis = 'x', scilimits = (0,0))
    ax.set_xlabel('Training Iteration', fontsize = 10)
    ax.legend(legend_space, fontsize = 10, loc = 'best')
    ax.grid(ls = ':', alpha = 0.5, color = 'grey')
    ax.set_title('Validation Loss', fontsize = 10)
    ax.set_ylabel('Cross Entropy Loss', fontsize = 10)
    ax = fig.add_subplot(1,2,2)
    for i in range(3):
        x_index = np.multiply(range(np.shape(val_stat_mean[i])[0]),300)
        ax.plot(x_index, val_stat_mean[i][:,1], color = color_space[i], ls = '-', lw = 2)
    ax.ticklabel_format(style = 'sci', axis = 'x', scilimits = (0,0))    
    ax.legend(legend_space, fontsize = 10, loc = 'best')
    ax.grid(ls = ':', alpha = 0.5, color = 'grey')
    ax.set_title('Validation Accuracy', fontsize = 10)
    ax.set_ylabel('Accuracy', fontsize = 10)
    ax.set_xlabel('Training Iteration', fontsize = 10)
    ax_global.set_title('Supervised Classification Result on CIFAR-10 Validation Data\n', fontsize = 12)
           
    
def Calc_Mean(index, train_stat):
    stat_mean = []
    for i,j in enumerate(index):
        if i < np.shape(index)[0]-1:
            stat_mean.append(np.mean(train_stat[index[i]:index[i+1],:],axis = 0))
    return np.array(stat_mean)
    
    
    
    

    
    
    
