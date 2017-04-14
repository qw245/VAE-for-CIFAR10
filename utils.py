import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
slim = tf.contrib.slim


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], size[2]))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img


def lrelu(x, leaky=0.2):
    return tf.maximum(x, leaky*x)

def encoder(input_tensor, output_size, size):
    '''Create encoder network.

    Args:
        input_tensor: a batch of flattened images [batch_size, nr*nc]
        output_size: dimensionality of hidden layer
        size: the size of image
    Returns:
        A tensor that expresses the encoder network
    '''
    with tf.variable_scope("inference"):
        # net = tf.reshape(input_tensor, [-1, size[2], size[0], size[1]])
        net = tf.reshape(input_tensor, [-1, size[0], size[1], size[2]])
        # net = tf.transpose(net, (0, 2, 3, 1))

        ## default
        # net = layers.conv2d(net, 32, 5, stride=2)
        # net = layers.conv2d(net, 64, 5, stride=2)
        # net = layers.conv2d(net, 128, 5, stride=1, padding='SAME')
        # net = layers.dropout(net, keep_prob=0.9)
        #  ############################################################
        #  net = layers.conv2d(net, 128, 5, stride=2, padding='SAME')
        #  net = layers.conv2d(net, 64, 5, stride=2, padding='SAME')
        #  net = layers.conv2d(net, 32, 5, stride=2, padding='VALID')

        net = slim.conv2d(net, 16, [5, 5], padding='SAME', stride=2, activation_fn=tf.nn.relu)  # bs x 14 x 14 x
        net = slim.conv2d(net, 32, [5, 5], padding='SAME', stride=2, activation_fn=tf.nn.relu)  # bs x 7 x 7 x
        net = slim.conv2d(net, 64, [5, 5], padding='VALID', stride=1, activation_fn=tf.nn.relu)  # bs x 2 x 2 x
        net = slim.conv2d(net, 1, [2, 2], padding='SAME', stride=1, activation_fn=tf.nn.sigmoid)
        # # net = slim.conv2d(net, 64, [5, 5], padding='SAME', stride=1, normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        #net = slim.flatten(net)
        #return slim.fully_connected(net, output_size, activation_fn=tf.nn.sigmoid)
        return tf.reshape(net, [-1, 1])


def discriminator(input_tensor, size):
    '''Create a network that discriminates between images from a dataset and
    generated ones.

    Args:
        input: a batch of real images [batch, height, width, channels]
    Returns:
        A tensor that represents the network
    '''

    return encoder(input_tensor, 1, size)


def decoder(input_tensor, output_size):
    '''Create decoder network.

        If input tensor is provided then decodes it, otherwise samples from
        a sampled vector.
    Args:
        input_tensor: a batch of vectors to decode
        output_size : the dimensionality of image

    Returns:
        A tensor that expresses the decoder network
    '''

    with tf.variable_scope("generator"):
        net = tf.expand_dims(input_tensor, 1)
        net = tf.expand_dims(net, 1)  # bs x 1 x 1 x 128
        # net = layers.conv2d_transpose(net, 128, 5, padding='VALID')
        # net = layers.conv2d_transpose(net, 64, 5, padding='VALID')
        # net = layers.conv2d_transpose(net, 32, 5, stride=2)
        # net = layers.conv2d_transpose(net, 1, 5, stride=2, activation_fn=None)
        #  net = layers.flatten(net)

        # MNIST
        net = slim.conv2d_transpose(net, 64, [5, 5], padding='SAME', stride=2, activation_fn=tf.nn.relu)  # bs x 2 x 2 x 64
        net = slim.conv2d_transpose(net, 32, [5, 5], padding='VALID', stride=2, activation_fn=tf.nn.relu)  # bs x 7 x 7 x 32
        net = slim.conv2d_transpose(net, 16, [5, 5], padding='SAME', stride=2, activation_fn=tf.nn.relu)  # bs x 14 x 14 x 16
        net = slim.conv2d_transpose(net, 1, [5, 5], padding='SAME', stride=2, activation_fn=tf.nn.sigmoid)  # bs x 28 x 28 x 1
        # net = slim.conv2d_transpose(net, 128, [5, 5], padding='SAME', stride=1, activation_fn=tf.nn.relu)
        # net = slim.conv2d_transpose(net, 64, [5, 5], padding='SAME', stride=2, activation_fn=tf.nn.relu)
        # net = slim.conv2d_transpose(net, 32, [5, 5], padding='SAME', stride=2, activation_fn=tf.nn.relu)
        # net = slim.conv2d_transpose(net, 1, [5, 5], padding='SAME', stride=1, activation_fn=tf.nn.sigmoid)
        # net = slim.conv2d_transpose(net, 3, [5, 5], padding='SAME', stride=2, activation_fn=tf.nn.sigmoid)
        #net = slim.flatten(net)
        #x = slim.fully_connected(net, output_size, activation_fn=tf.nn.sigmoid)
        x = net
        return x

