#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sat Jun 03

@author: jacob

Applying 'Alexnet' to Category Cervical Cancer Dataset classification task.
References:
	- Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet
	Classification with Deep Convolutional Neural Networks. NIPS, 2012.
Links:
	- [AlexNet Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
	- [Cervical Cancer Dataset](https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/data)
"""

from __future__ import division, print_function, absolute_import

import os
from glob import glob

import tflearn
from tflearn.data_utils import image_preloader
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

USER_ROOT = os.path.dirname(os.path.realpath(__file__))
TYPES = ['Type_1','Type_2','Type_3']

def generate_dataset_path():
	with open(os.path.join(USER_ROOT, 'dataset.txt'), 'w') as output:
		for type in TYPES:
			files = glob(os.path.join(USER_ROOT, 'pre_processed', type, "*.jpg"))
			for line in files:
				output.write(line + '\t' + str(int(type[-1])-1) + '\n')


if __name__ == '__main__':
	generate_dataset_path()

	X, Y = image_preloader(os.path.join(USER_ROOT, 'dataset.txt'), image_shape=(200, 200),
						   mode='file', categorical_labels=True, normalize=True)

	# Real-time data preprocessing
	img_prep = ImagePreprocessing()
	img_prep.add_featurewise_zero_center()
	img_prep.add_featurewise_stdnorm()

	# Real-time data augmentation
	img_aug = ImageAugmentation()
	img_aug.add_random_flip_leftright()
	img_aug.add_random_rotation(max_angle=25.)

	# Building 'AlexNet'
	network = input_data(shape=[None, 200, 200, 3],
                     	 data_preprocessing=img_prep, data_augmentation=img_aug)
	network = conv_2d(network, 96, 11, strides=4, activation='relu')
	network = max_pool_2d(network, 3, strides=2)
	network = local_response_normalization(network)
	network = conv_2d(network, 256, 5, activation='relu')
	network = max_pool_2d(network, 3, strides=2)
	network = local_response_normalization(network)
	network = conv_2d(network, 384, 3, activation='relu')
	network = conv_2d(network, 384, 3, activation='relu')
	network = conv_2d(network, 256, 3, activation='relu')
	network = max_pool_2d(network, 3, strides=2)
	network = local_response_normalization(network)
	network = fully_connected(network, 4096, activation='tanh')
	network = dropout(network, 0.5)
	network = fully_connected(network, 4096, activation='tanh')
	network = dropout(network, 0.5)
	network = fully_connected(network, 3, activation='softmax')
	network = regression(network, optimizer='adam',
						 loss='categorical_crossentropy', learning_rate=0.001)

	# Training
	model = tflearn.DNN(network, tensorboard_verbose=0)
	model.fit(X, Y, n_epoch=10, validation_set=0.1, shuffle=True, show_metric=True,
			  batch_size=64, snapshot_step=200, snapshot_epoch=False)