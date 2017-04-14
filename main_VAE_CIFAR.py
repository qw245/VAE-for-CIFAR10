'''TensorFlow implementation of http://arxiv.org/pdf/1312.6114v10.pdf'''

from __future__ import absolute_import, division, print_function

import math
import os

import numpy as np
import scipy.misc
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.contrib.framework import arg_scope
from scipy.misc import imsave
from tensorflow.examples.tutorials.mnist import input_data
from utils import merge

# from progressbar import ETA, Bar, Percentage, ProgressBar

from vae import VAE
from gan import GAN

flags = tf.flags
logging = tf.logging


def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

file = "cifar-10-batches-py/data_batch_1"
images_1 = unpickle(file)
images_t = images_1['data']

file = "cifar-10-batches-py/data_batch_2"
images_2 = unpickle(file)
images_t = np.concatenate((images_t, images_2['data']), axis=0)

file = "cifar-10-batches-py/data_batch_3"
images_3 = unpickle(file)
images_t = np.concatenate((images_t, images_3['data']), axis=0)

file = "cifar-10-batches-py/data_batch_4"
images_4 = unpickle(file)
images_t = np.concatenate((images_t, images_4['data']), axis=0)

file = "cifar-10-batches-py/data_batch_5"
images_5 = unpickle(file)
images_train = np.concatenate((images_t, images_5['data']), axis=0)
images_train.astype(float)
images_t = images_train / 255.0

file = "cifar-10-batches-py/test_batch"
im_test = unpickle(file)['data']
im_test.astype(float)
images_test = im_test / 255.0


params = {
    "batch_size": 128,
    "max_epoch": 50,
    "learning_rate": 2e-4,
    "working_directory": "",
    "hidden_size": 128,
    "model": "vae"
}

# data_directory = os.path.join(params["working_directory"], "MNIST")
# if not os.path.exists(data_directory):
#     os.makedirs(data_directory)

n_samples = 50000
nr = 32
nc = 32
nch = 3

params["updates_per_epoch"] = int(n_samples / params["batch_size"])
params["image_size"] = [nr, nc, nch]


im_base_train = images_t[0:params['batch_size'], :]
im_base_test = images_test[0:params['batch_size'], :]
reshaped_im_train = np.transpose(im_base_train.reshape(params['batch_size'], nch, nr, nc), (0, 2, 3, 1))
reshaped_im_test = np.transpose(im_base_test.reshape(params['batch_size'], nch, nr, nc), (0, 2, 3, 1))
imsave("results/base_train.jpg", np.squeeze(merge(reshaped_im_train[:64], [8, 8, nch])))
imsave("results/base_test.jpg", np.squeeze(merge(reshaped_im_test[:64], [8, 8, nch])))


assert params["model"] in ['vae', 'gan']  # Check weather FLAGS.model is correctly defined
if params["model"] == 'vae':
    model = VAE(params["hidden_size"], params["batch_size"], params["learning_rate"], params["image_size"])
elif params["model"] == 'gan':
    model = GAN(params["hidden_size"], params["batch_size"], params["learning_rate"])

config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1),
                        log_device_placement=True)
config.gpu_options.allow_growth = True
init_op = tf.global_variables_initializer()
with tf.Session(config=config) as sess:

    saver = tf.train.Saver()
    sess.run(init_op)

    for epoch in range(params["max_epoch"]):
        training_loss = 0.0
        training_g_loss = 0.0

        # pro_bar = ProgressBar()
        # for i in pro_bar(range(FLAGS.updates_per_epoch)):
        for idx in range(params["updates_per_epoch"]):
            images = images_t[params['batch_size'] * idx:params['batch_size'] * (idx + 1), :]
            loss_value = model.update_params(images)
            training_loss += loss_value

        training_loss /= (params["updates_per_epoch"] * params["batch_size"])

        # print("%d epoch: DIS Loss %f, GEN Loss %f " % (epoch+1, training_loss, training_g_loss))
        print("%d epoch: Loss %f" % (epoch + 1, training_loss))

        model.generate_and_save_images(params["batch_size"], params["working_directory"], im_base_train, im_base_test,
                                       [nr, nc, nch], epoch)
        # saver_path = saver.save(sess, os.getcwd() + "/training/train", global_step=epoch)
