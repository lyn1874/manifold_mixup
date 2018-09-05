# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 12:07:49 2018

@author: s161488
"""

import tensorflow as tf
import numpy as np
import os
from resnet_simple_basic import build_tower as build_tower_basic
from resnet_simple_preact import build_tower as build_tower_preact
from Loss import train_op_batchnorm, Calc_Loss, train_op_batchnorm_no_loss_average
from cifar10 import Read_Data_from_Record
LABLE_SPACE = ["airplane", "automobile", "bird", "cat", "deer","dog","frog","horse","ship","trunk"]

def Check_ResNet():
    """This Func is for checking the dimension of input and output"""
    alpha = 2
    images = tf.constant(0.3, shape = [5,32,32,3], dtype = tf.float32)
    labels = tf.constant(1, shape = [5], dtype = tf.int64)
    input_lambda = np.random.beta(alpha, alpha)   
    NUM_CLASS = 10
    print(input_lambda)
    phase_train = tf.placeholder(tf.bool, name = 'phase_train')
    logits, target = build_tower_preact(images, labels, input_lambda, NUM_CLASS, phase_train, mixup_option = True)
    print("The final logits", logits)
    print("The final target", target)
    return logits, target
    
    
def Train(max_steps, batch_size, loss_average, MIXUP_OPTION, resnet_option):
    #--------Here lots of parameters need to be set------Or maybe we could set it in the configuration file-----#
    learning_rate = 0.001
    image_w, image_h, image_c = [32,32,3]
    decay_rate = 0.1
    decay_steps = (40000/batch_size)*100
    epsilon_opt = 0.001
    #resnet_ckpt_dir = "/zhome/1c/2/114196/Documents/SA"
    ckpt_dir = None
    MOVING_AVERAGE_DECAY=0.999
    alpha = 2
    val_step_size = 100
    NUM_CLASS = 10       
    FLAG_PRETRAIN = False
    logs_path = "/scratch/mixup/"
    model_dir = os.path.join(logs_path, 'Iter_%d_Batch_Size_%d_Loss_Average_%s_%s'%(max_steps, batch_size, loss_average, MIXUP_OPTION)) #Remember to modify this
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    checkpoint_path = os.path.join(model_dir,'model.ckpt')    
    TRAINING_PATH = ["/scratch/mixup/train.tfrecords"]
    VALIDATION_PATH = ["/scratch/mixup/validation.tfrecords"]
    with tf.Graph().as_default():
        images_train = tf.placeholder(tf.float32, [batch_size, image_w,image_h,image_c])
        labels_train = tf.placeholder(tf.int64, [batch_size])
        phase_train = tf.placeholder(tf.bool, shape = None)
        Selec_Layer_Index = tf.placeholder(tf.int64)
        input_lambda_tensor = tf.placeholder(tf.float32, shape = [1])
        global_step = tf.train.get_or_create_global_step()
        image_batch_train, label_batch_train = Read_Data_from_Record(TRAINING_PATH, batch_size, augmentation = True, shuffle = True)
        image_batch_val, label_batch_val = Read_Data_from_Record(VALIDATION_PATH, batch_size, augmentation = False, shuffle = True)        

        if resnet_option == "preact":
            logits, target = build_tower_preact(images_train, labels_train, input_lambda_tensor, NUM_CLASS, phase_train, Selec_Layer_Index)
            print("I am using the pre-activation resnet structure")
        elif resnet_option == "basic":
            logits, target = build_tower_basic(images_train, labels_train, input_lambda_tensor, NUM_CLASS, phase_train, Selec_Layer_Index)
            print("I am using the basic resnet structure")
        print("------The output logits and target------------", logits, target)
        Data_Error_Loss, Init_Accu, Reweight_Accu = Calc_Loss(logits, labels_train, target, input_lambda_tensor)
        var_train = tf.trainable_variables()

        Total_Loss = tf.add_n([Data_Error_Loss, tf.add_n([tf.nn.l2_loss(v) for v in var_train if ('bias' not in v.name)])*0.001], name = "Total_Loss")
        if loss_average is True:
            train = train_op_batchnorm(total_loss = Total_Loss, global_step = global_step, initial_learning_rate = learning_rate, lr_decay_rate = decay_rate, decay_steps = decay_steps,
                                       epsilon_opt = epsilon_opt, var_opt = var_train, MOVING_AVERAGE_DECAY = MOVING_AVERAGE_DECAY)
        else:
            train = train_op_batchnorm_no_loss_average(total_loss = Total_Loss, global_step = global_step,initial_learning_rate = learning_rate,
                                                       lr_decay_rate = decay_rate, decay_steps = decay_steps, 
                                                       epsilon_opt = epsilon_opt, var_opt = var_train, MOVING_AVERAGE_DECAY = MOVING_AVERAGE_DECAY)
        summary_op = tf.summary.merge_all()
            
        if FLAG_PRETRAIN is False:
            saver = tf.train.Saver(tf.global_variables(), max_to_keep = 2)
            summary_op = tf.summary.merge_all() #now I couldn't find the pre-trained resnet weight
        else:
            saver = tf.train.Saver(tf.global_variables())                
            
        print("\n =====================================================")
        with tf.Session() as sess:
            if FLAG_PRETRAIN is False:
                sess.run(tf.variables_initializer(tf.global_variables()))
                sess.run(tf.local_variables_initializer())
            else:
                sess.run(tf.variables_initializer(tf.global_variables()))
                sess.run(tf.local_variables_initializer())
                saver.restore(sess, ckpt_dir)      
                
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord = coord)
                
            Total_Train_Stat = np.zeros([max_steps, 3]) #loss, init_accu, reweight_accu
            Total_Val_Stat = np.zeros([max_steps, 2]) #loss, init_accu, reweight_accu
            #train_writer = tf.summary.FileWriter(model_dir, sess.graph)
            print("-------------------------Start Training-------------------------------------")
            for step in range(max_steps):
                if MIXUP_OPTION == "pure_resnet":
                    selected_layer_index = 4
                    input_lambda = 1.0
                elif MIXUP_OPTION == "input_mixup":
                    selected_layer_index = 0
                    input_lambda = np.random.beta(alpha, alpha)   
                elif MIXUP_OPTION == "manifold_mixup":
                    selected_layer_index = np.random.randint(0,3,[1])[0]
                    input_lambda = np.random.beta(alpha, alpha)   
                
                
                #print("----------------Input lambda is--------", input_lambda)
                image_train_batch, label_train_batch = sess.run([image_batch_train, label_batch_train])

                feed_dict = {images_train:image_train_batch,
                             labels_train:label_train_batch,
                             phase_train:True,
                             Selec_Layer_Index:selected_layer_index,
                             input_lambda_tensor: [input_lambda]}
                _, Data_Error_Loss_Train, Init_Accu_Train, Reweight_Accu_Train = sess.run([train,Data_Error_Loss, Init_Accu, Reweight_Accu], feed_dict = feed_dict)
                        
                Total_Train_Stat[step,:] = [Data_Error_Loss_Train, Init_Accu_Train, Reweight_Accu_Train]
                if step % val_step_size == 0:
                    print("start validating .......")
                    _val_loss = []
                    _val_init_accu = [] 
                    #_val_reweight_accu = []
                    for val_step in range(int(10000//batch_size)):
                        image_val_batch, label_val_batch = sess.run([image_batch_val, label_batch_val])
                        fetches_valid = [Data_Error_Loss, Init_Accu]
                        feed_dict_valid = {images_train: image_val_batch,
                                           labels_train: label_val_batch,
                                           phase_train: False,
                                           Selec_Layer_Index: 3,
                                           input_lambda_tensor:[1.0]}
                        _val_single_loss, _val_single_init_accu = sess.run(fetches = fetches_valid,feed_dict = feed_dict_valid)
                        _val_loss.append(_val_single_loss)
                        _val_init_accu.append(_val_single_init_accu)
                        
                    
                    Total_Val_Stat[step//val_step_size,:] = [np.mean(_val_loss), np.mean(_val_init_accu)]
                    print("validation", step, Total_Val_Stat[step//val_step_size,:])
                
                if max_steps % 1000 == 0 or step == (max_steps-1):
                    saver.save(sess,checkpoint_path,global_step = step)
                if step == (max_steps-1):
                    np.save(os.path.join(model_dir, 'trainstat'), Total_Train_Stat)
                    np.save(os.path.join(model_dir, 'valstat'), Total_Val_Stat)
                    
            
            coord.request_stop()
            coord.join(threads)
            
            
def Test_Mixup_ResNet():
    image_w, image_h, image_c = [32,32,3]
    path_mom = "/scratch/mixup"
    path_son_all = os.listdir(path_mom)
    path_selec = [v for v in path_son_all if 'Iter' in v and '_preactivation' not in v]
    print(path_selec)
    MOVING_AVERAGE_DECAY=0.999
    NUM_CLASS = 10       
    TEST_PATH = ["/scratch/mixup/eval.tfrecords"]
    batch_size = 1
    num_image = 10000
    with tf.Graph().as_default():
        images_test = tf.placeholder(tf.float32, [batch_size, image_w,image_h,image_c])
        labels_test = tf.placeholder(tf.int64, [batch_size])
        phase_train = tf.placeholder(tf.bool, shape = None)
        Selec_Layer_Index = tf.placeholder(tf.int64)
        input_lambda_tensor = tf.placeholder(tf.float32, shape = [1])
        image_batch_te, label_batch_te = Read_Data_from_Record(TEST_PATH, batch_size, augmentation = False, shuffle = False)        
        logits, target = build_tower_basic(images_test, labels_test, input_lambda_tensor, NUM_CLASS, phase_train, Selec_Layer_Index)
        Data_Error_Loss, Init_Accu, Reweight_Accu = Calc_Loss(logits, labels_test, target, input_lambda_tensor)
        var_train = tf.trainable_variables()
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variable_averages.apply(var_train)
        variables_to_restore = variable_averages.variables_to_restore(tf.moving_average_variables())
        saver = tf.train.Saver(variables_to_restore)    
        for single_ckpt in path_selec:
            ckpt_dir = os.path.join(path_mom, single_ckpt)
            print("\n================================")
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(ckpt_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print("restore parameter from ", ckpt.model_checkpoint_path)
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord = coord)
                Total_Test_Stat = np.zeros([num_image, 2]) #loss, init_accu, reweight_accu
                print("-------------------------Start Training-------------------------------------")
                for step in range(num_image):
                    selected_layer_index = 3
                    input_lambda = 1
					#print("----------------Input lambda is--------", input_lambda)
                    image_test_batch, label_test_batch = sess.run([image_batch_te, label_batch_te])
                    feed_dict = {images_test:image_test_batch,	
								labels_test:label_test_batch,
								phase_train:False,
								Selec_Layer_Index:selected_layer_index,
								input_lambda_tensor: [input_lambda]}
                    Data_Error_Loss_Train, Init_Accu_Train = sess.run([Data_Error_Loss, Init_Accu], feed_dict = feed_dict)
                    Total_Test_Stat[step,:] = [Data_Error_Loss_Train, Init_Accu_Train]
                print("-------Test Error, Test Accu------------",np.mean(Total_Test_Stat,axis = 0))
                coord.request_stop()
                coord.join(threads)                               
                                                                     
        
                                                                     
                                                                     
        


    

    
