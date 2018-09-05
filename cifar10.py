# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CIFAR-10 data set.

See http://www.cs.toronto.edu/~kriz/cifar.html.
"""
import tensorflow as tf

HEIGHT = 32
WIDTH = 32
DEPTH = 3
TARGET_IM_H = 32
TARGET_IM_W = 32


def Read_Data_from_Record(filename, batch_size, augmentation, shuffle = True):
    """This function is utilized to read the data which is saved in the tf.record file
    Args:
        batch_size: batch size determine how many images are utilized for one iteration of training
    Returns:
        The augmentated and cropped images
        image_aug: [Batch_size, Image_height, Image_width, 3]
        Label_aug: [Batch_size, Image_height, Image_width, 1]
        Edge_aug : [Batch_size, Image_height, Image_width, 1]
        Image Index: [Batch_Size,1]
        here image_height and image_width are fixed to be 512
    """
    
    #filename = ["/zhome/1c/2/114196/Documents/Master_Project/Gland/glanddata.tfrecord"]
    filename_queue = tf.train.string_input_producer(filename)
    reader = tf.TFRecordReader()
    #serialized_example is a tensor of type string
    _, serialized_example = reader.read(filename_queue)
    #The serialized example is converted back to actual values, we need to describe the format of the objects to be returned
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    image = tf.decode_raw(features['image'], tf.uint8)
    image.set_shape([DEPTH * HEIGHT * WIDTH])

    # Reshape from [depth * height * width] to [depth, height, width].
    image = tf.cast(
        tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
        tf.float32)
    label = tf.cast(features['label'], tf.int32)
    #-----------------From Here, We need to do augmentation for the data--------------------#
    image_aug = Random_Flip_Rotate_Crop(image, augmentation)
    #min_fraction_of_examples_in_queue = 0.4
    #min_queue_examples = int(NUM_EXAMPLES_FOR_TRAIN * min_fraction_of_examples_in_queue)
    min_queue_examples = 100    
    print('Filling queue with %d Gland images bfore starting to train. this will take a few minutes'%min_queue_examples)
    
    return _generate_image_and_label_batch(image_aug, label, min_queue_examples,batch_size, shuffle = shuffle)
                        
                        
                        
def Random_Flip_Rotate_Crop(image_single, aug):
    """This function is utilized to conduct augmentation for the data. To make sure the image and the ground truth are
    rotated at the same time. First we perform per channel zero mean for the image. Then we concate the image, label, edge,
    and bdbox together. Then we conduct the random flip up and down, random flip left to right   
    
    Args:
        image_single: A tensor. [Image_Height, Image_Width, 3] tf.int32    
    Returns:
        image: a tensor, [TARGET_IM_H, TARGET_IM_W, 3], tf.float32
    Operations:
        1. random_brightness
        2. random flip up down
        3. random rotate
        4. crop and resize image w.r.t center of the image
    """
    if aug is True:
        image = tf.image.random_brightness(tf.cast(image_single, tf.float32), max_delta = 10.0)
        image = tf.image.random_flip_up_down(image)
        k = tf.random_uniform(shape = [1], minval = 1, maxval = 5, dtype = tf.int32)
        image = tf.image.rot90(image, k = k[0])
        image = tf.image.resize_image_with_crop_or_pad(image, TARGET_IM_H, TARGET_IM_W)
        print("---------------------I am using augmentation skills---------------------------")
    else:
        image = tf.cast(image_single,tf.float32)
        print("---------------------I am not using any augmentation skills -------------------")
    return image




def _generate_image_and_label_batch(image,label, min_queue_examples,batch_size,shuffle):
#def _generate_image_and_label_batch(image,label,min_queue_examples,batch_size,shuffle):
    """Construct a queued batch of images and batches.
    
    Args:
        image: 3-D Tensor of [height, width, 3] of type.float32.
        label: 3-D Tensor of type.int64
        min_queue_examples: int32, minimum number of samples to retain
                        in the queue that provides of batches of examples.
        batch_size: Number of images per batch.
        shuffle: boolean indicating whether to use a shuffling queue.
    Returns:
        images: Images. 4D tensor of [batch_size, height, width, 3] size.
        labels: Labels. 4D tensor of [batch_size, height, width, 1] size.
    """
    num_preprocess_threads = 1
    if shuffle:
        image_batch, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        image_batch, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)
        print("ohhh, the images are not shuffled")

    return image_batch,label_batch

