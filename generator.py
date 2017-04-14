import os
from scipy.misc import imsave
from utils import merge
import numpy as np


class Generator(object):

    def update_params(self, input_tensor):
        '''Update parameters of the network

        Args:
            input_tensor: a batch of flattened images

        Returns:
            Current loss value
        '''
        raise NotImplementedError()

    def generate_and_save_images(self, num_samples, directory, img_base_train, img_base_test, size, epoch):
        '''Generates the images using the model and saves them in the directory

        Args:
            num_samples: number of samples to generate
            directory: a directory to save the images
            img_base_train: the train image to be reconstructed
            img_base_test: the test image to be reconstructed
            size: the size of image
            epoch: the number of epoch
        '''
        nr, nc, nch = size[0], size[1], size[2]
        img_sample = self.sess.run(self.sampled_tensor)
        # rec_train = self.sess.run(self.recons_tensor, feed_dict={self.input_tensor: img_base_train})
        # rec_test = self.sess.run(self.recons_tensor, feed_dict={self.input_tensor: img_base_test})
        # img_sample = np.transpose(img_sample.reshape(num_samples, nch, nr, nc), (0, 2, 3, 1))
        img_sample = img_sample.reshape(num_samples, nr, nc, 1)
        # img_rec_train = np.transpose(rec_train.reshape(num_samples, nch, nr, nc), (0, 2, 3, 1))
        # img_rec_test = np.transpose(rec_test.reshape(num_samples, nch, nr, nc), (0, 2, 3, 1))

        # print("train error: %f  test error: %f  " % (np.mean(np.square(rec_train - img_base_train)),
        #                                              np.mean(np.square(rec_test - img_base_test))))

        imsave("results/sample/x_sample/" + str(epoch) + "_X_sample.jpg",
               np.squeeze(merge(img_sample[:64], [8, 8, size[2]])))
        # imsave("results/recons/x_train/" + str(epoch) + "_X_train.jpg",
        #        np.squeeze(merge(img_rec_train[:64], [8, 8, size[2]])))
        # imsave("results/recons/x_test/" + str(epoch) + "_X_test.jpg",
        #        np.squeeze(merge(img_rec_test[:64], [8, 8, size[2]])))

        # for k in range(img_sample.shape[0]):
        #     imgs_folder = os.path.join(directory, 'imgs')
        #     if not os.path.exists(imgs_folder):
        #         os.makedirs(imgs_folder)
        #
        #     imsave(os.path.join(imgs_folder, '%d.jpg') % k,
        #            img_sample[k].reshape(28, 28))

