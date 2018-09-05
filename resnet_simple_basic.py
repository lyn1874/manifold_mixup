# -*- coding: utf-8 -*-
"""
Rewrite the ResNet-18 based on https://github.com/dalgu90/resnet-18-tensorflow

author: s161488
Time: Sep-03-2018
"""
import tensorflow as tf
import numpy as np


def build_tower(images, labels, input_lambda, NUM_CLASS, is_training, Selec_Layer_Index):
    """
    Selec_Layer_Index: tensor, dtype: tf.int64"""
#TODO: A MIX UP OPTION NEEDS TO BE DEFINED SO THAT WE CAN MAKE SURE THE MIXUP IS NOT USED IN VALIDATION PERIOD    
    print('Building model')
    # filters = [128, 128, 256, 512, 1024]
    filters = [64, 64, 128, 256, 512]
    kernels = [3, 3, 3, 3, 3]
    strides = [1, 0, 2, 2, 2]
    print("the index for the selected layer", Selec_Layer_Index)
    # conv1
    x = images
    target = labels
    results = tf.cond(tf.equal(Selec_Layer_Index,tf.constant(0, dtype = tf.int64)),
                      lambda: mix_up_process(x, target, input_lambda, True),
                      lambda: mix_up_process(x, target, input_lambda, False))
    x, target = results
    print('\tBuilding unit: conv1')
    with tf.variable_scope('conv1'):
        x = _conv(x, kernels[0], filters[0], strides[0], bias_stat = False)
        x = _bn(x, is_training, scope = 'initial_bn')
        x = _relu(x)
        #x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

    # conv2_x
        
    x = _residual_block(x, is_training = is_training, name='conv2_1')
    x = _residual_block(x, is_training = is_training, name='conv2_2')
    results = tf.cond(tf.equal(Selec_Layer_Index,tf.constant(1, dtype = tf.int64)),
                      lambda: mix_up_process(x, target, input_lambda, True),
                      lambda: mix_up_process(x, target, input_lambda, False))
    x, target = results
                        
    # conv3_x
    x = _residual_block_first(x, filters[2], strides[2], is_training = is_training, name='conv3_1')
    x = _residual_block(x, is_training = is_training, name='conv3_2')
    results = tf.cond(tf.equal(Selec_Layer_Index,tf.constant(2, dtype = tf.int64)),
                      lambda: mix_up_process(x, target, input_lambda, True),
                      lambda: mix_up_process(x, target, input_lambda, False))
    x, target = results
        # conv4_x
    x = _residual_block_first(x, filters[3], strides[3], is_training = is_training, name='conv4_1')
    x = _residual_block(x, is_training = is_training, name='conv4_2')

    # conv5_x
    x = _residual_block_first(x, filters[4], strides[4], is_training = is_training, name='conv5_1')
    x = _residual_block(x, is_training = is_training, name='conv5_2')

    # Logit
    with tf.variable_scope('logits') as scope:
        print('\tBuilding unit: %s' % scope.name)
        #print(x)
#        x = tf.reduce_mean(x, [1, 2])
        print("Before average pooling", x)
        x = tf.nn.avg_pool(x,ksize = (1,4,4,1), strides = (1,1,1,1), padding = 'VALID')
        print("After average pooling",x)
        x = _fc(x, NUM_CLASS)
        print("After fully connected layer",x)
        x = tf.reshape(x,[-1,NUM_CLASS])
    logits = x

    return logits, target
        
def _residual_block_first(x, out_channel, strides, is_training = True, name="unit"):
    """
        This new version of resnet basically add the mixup option
        target: the label
        input_lambda: the lambda which will be determined from begining
        mixup: is a random selected option, logical, True or False
    """
    in_channel = x.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
         print('\tBuilding residual unit: %s' % scope.name)
         # Shortcut connection
         if in_channel == out_channel:
            if strides == 1:
                shortcut = tf.identity(x)
            else:
                shortcut = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
         else:
            shortcut = _conv(x, 1, out_channel, strides, name='shortcut')
        # Residual
         x = _conv(x, 3, out_channel, strides, name = 'conv_1')
         x = _bn(bias_input = x, is_training = is_training, scope = "bn_1")
         x = _relu(x, name = 'relu_1')
         x = _conv(x, 3, out_channel, 1, name = 'conv_2')
         x = _bn(bias_input = x, is_training = is_training, scope = "bn_2")
        # Merge
         x = x + shortcut
         x = _relu(x, name = 'relu_2')
    #print("-------------After residual block first x------",x)
    return x


def _residual_block(x, is_training = True, name="unit"):
    """This new version of residual block
    target: the ground truth label
    input_lambda: the defined input_lambda
    mixup: the mixup option, logical, True or False
    """
    num_channel = x.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
        print('\tBuilding residual unit: %s' % scope.name)
        # Shortcut connection
        # Residual¨
        shortcut = x
        x = _conv(x,3, num_channel, 1, name = 'conv_1')
        x = _bn(x, is_training, scope = "bn_1")
        x = _relu(x, name = 'relu_1')
        x = _conv(x, 3, num_channel, 1, name = 'conv_2')
        x = _bn(x, is_training, scope = "bn_2")
        x = x + shortcut
        
        x = _relu(x, name = 'relu_2')
    #print("-------------After residual block x------",x)

    return x

    # Helper functions(counts FLOPs and number of weights)
def _conv(x, filter_size, out_channel, stride, pad="same", name="conv", bias_stat = True):
    x = tf.layers.conv2d(x, out_channel, kernel_size = filter_size, strides = stride, padding = pad,
                         activation = tf.nn.relu, use_bias = bias_stat, kernel_initializer = tf.truncated_normal_initializer(stddev = 0.01),
                         name = name)    
    #print(x)
    return x    

def _fc(x, NUM_CLASS, name="fc"):
    x = tf.layers.dense(x, units = NUM_CLASS, activation = None, use_bias = True, kernel_initializer = tf.truncated_normal_initializer(stddev = 0.01), name = name)
    return x    

#def _bn(bias_input, is_training, scope):
#    return tf.cond(is_training,
#                   lambda: tf.contrib.layers.batch_norm(bias_input, is_training=True, center=False, scope=scope),
#                   lambda: tf.contrib.layers.batch_norm(bias_input, is_training=False,center=False, reuse = True, scope=scope))

def _bn(bias_input, is_training, scope):
    return tf.layers.batch_normalization(inputs = bias_input, momentum = 0.997, epsilon = 1e-5, center = True, scale = True, training = is_training,
                                         fused = True, name = scope)
def _relu(x, name="relu"):
    x = tf.nn.relu(x,name = name)
    return x

#def mix_up_process(x_input, target_input, lambda_input, mixup_option):
#    """
#    Args:
#        x_input: [Batch_Size, F_H, F_W, N_C]
#        target_input: [Batch_Size]
#    """
#    if mixup_option is False:
#        result_output = [x_input, target_input]
#        indices = None
#    else:
#        num_image = x_input.shape.as_list()[0]
#        indices = np.random.choice(range(num_image),[num_image],replace = False)
#        print(indices)
#        x_input_shuffle = tf.gather(x_input, indices, axis = 0)
#        x_output = tf.multiply(x_input,lambda_input)+tf.multiply(x_input_shuffle,(1-lambda_input))
#        y_output = tf.gather(target_input, indices, axis = 0)
#        result_output = [x_output, y_output]
#    
#    return result_output, indices

def mix_up_process(x_input, target_input, lambda_input, mixup_option):
    """
    Args:
        x_input: [Batch_Size, F_H, F_W, N_C]
        target_input: [Batch_Size]
    """
    if mixup_option is False:
        result_output = [x_input, target_input]
        indices = tf.range(5)
    else:
        num_image = x_input.shape.as_list()[0]
        origi_index = tf.range(num_image)
        indices = tf.random_shuffle(origi_index)        
        x_input_shuffle = tf.gather(x_input, indices, axis = 0)
        x_output = tf.multiply(x_input,lambda_input)+tf.multiply(x_input_shuffle,(1-lambda_input))
        y_output = tf.gather(target_input, indices, axis = 0)
        result_output = [x_output, y_output]
    
    return result_output

