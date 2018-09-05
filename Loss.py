# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 12:48:36 2018
This file includes the corresponding loss function and accuracy measurements and also the training method
@author: s161488
"""
import tensorflow as tf
import numpy as np

def Calc_Loss(logits, target_initial, target_reweight, lambda_input):
    """This function is for calculating the loss
    target_initial: shape = [batch_size], dtype = tf.int64
    target_reweight: shape = [batch_size], dtype = tf.int64
    logits: shape = [batch_size, Num_Class], dtype = tf.float32
    lambda_input: float
    """
    #Num_Class = logits.get_shape().as_list()[-1]
    init_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = target_initial, logits = logits, name = "cross_entropy"),name = "init_loss")
    reweight_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = target_reweight, logits = logits, name = "cross_entropy"), name = "re_loss")
    tf.add_to_collection('loss', init_loss)
    tf.add_to_collection('loss', reweight_loss)
    
    Total_Loss = lambda_input*init_loss+(1-lambda_input)*reweight_loss
    print(init_loss, reweight_loss, Total_Loss)
    correct_prediction_init = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,-1), target_initial,"init_accu"),tf.float32))
    correct_prediction_re = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,-1),target_reweight,"reweight_accu"),tf.float32))
    
    return Total_Loss[0], correct_prediction_init, correct_prediction_re
    
    
def train_op_batchnorm(total_loss, global_step,initial_learning_rate,lr_decay_rate, decay_steps, epsilon_opt, var_opt, MOVING_AVERAGE_DECAY):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    lr_decay_rate,
                                    staircase=False) #if staircase is True, then the division between global_step/decay_steps is a interger, 
    #otherwise it's not an interger. 
    tf.summary.scalar('learning_rate', lr)
    
    variables_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    #variables_averages_op = variables_averages.apply(var_opt)
    with tf.control_dependencies(update_ops):
        loss_averages_op = _add_loss_summaries(total_loss)
        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.AdamOptimizer(lr, epsilon = epsilon_opt)
            grads = opt.compute_gradients(total_loss, var_list = var_opt)
            
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
          
        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
        for grad, var in grads:
            print(grad)
                
        apply_gradient_op = opt.apply_gradients(grads, global_step = global_step)
        with tf.control_dependencies([apply_gradient_op]):
            train_op = variables_averages.apply(var_opt)
        
    return train_op  
    
    
def train_op_batchnorm_no_loss_average(total_loss, global_step,initial_learning_rate,lr_decay_rate, decay_steps, epsilon_opt, var_opt, MOVING_AVERAGE_DECAY):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    lr_decay_rate,
                                    staircase=False) #if staircase is True, then the division between global_step/decay_steps is a interger, 
    #otherwise it's not an interger. 
    tf.summary.scalar('learning_rate', lr)
    
    variables_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    #variables_averages_op = variables_averages.apply(var_opt)
    with tf.control_dependencies(update_ops):
        opt = tf.train.AdamOptimizer(lr, epsilon = epsilon_opt)
        grads = opt.compute_gradients(total_loss, var_list = var_opt)
        
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
          
        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
                
        apply_gradient_op = opt.apply_gradients(grads, global_step = global_step)
        with tf.control_dependencies([apply_gradient_op]):
            train_op = variables_averages.apply(var_opt)
        
    return train_op  
        
    
def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.
  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('loss')
  loss_averages_op = loss_averages.apply(losses+[total_loss])
  for l in losses + [total_loss]:
    tf.summary.scalar(l.op.name+'_raw', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))
  return loss_averages_op
    

    
    
