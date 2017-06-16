# -*- coding: utf-8 -*-

""" 
Created on Sat Jun 03

@author: jacob

inception_resnet_v2.
Applying 'inception_resnet_v2' to Cervical Cancer Dataset classification task.
References:
	Inception-v4, Inception-ResNet and the Impact of Residual Connections
	on Learning
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi.
Links:
	http://arxiv.org/abs/1602.07261
"""

from __future__ import absolute_import, division, print_function

import os
from glob import glob
import numpy as np
import pandas as pd

import tflearn
from tflearn.data_utils import image_preloader
import tflearn.activations as activations

# Data loading and preprocessing
from tflearn.activations import relu
from tflearn.data_utils import shuffle
from tflearn.layers.conv import avg_pool_2d, conv_2d, max_pool_2d
from tflearn.layers.core import dropout, flatten, fully_connected, input_data
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import batch_normalization
from tflearn.utils import repeat

USER_ROOT = os.path.dirname(os.path.realpath(__file__))
TYPES = ['Type_1','Type_2','Type_3','AType_1','AType_2','AType_3','NType_1','NType_2','NType_3']

def generate_dataset_path(filename, boolean):
	if boolean:
		with open(os.path.join(USER_ROOT, filename+'.txt'), 'w') as output:
			for type in TYPES:
				files = glob(os.path.join(USER_ROOT, 'pre_processed', type, "*.jpg"))
				for line in files:
					output.write(line + '\t' + str(int(type[-1])-1) + '\n')
	else:
		for i in range(8):
			with open(os.path.join(USER_ROOT, filename+str(i)+'.txt'), 'w') as output:
				files = glob(os.path.join(USER_ROOT, 'pre_processed', 'test_stg2', "*.jpg"))
				trunk = int(len(files)/8)
				if i == 7:
					for line in files[i*trunk:]:
						output.write(line + '\t0\n')
				else:
					for line in files[i*trunk:(i+1)*trunk]:
						output.write(line + '\t0\n')

def split_training_testing(X, y):
	# split the data into training and testing for hold-out testing    
	n_rows, _, _, _ = np.shape(X)
		
	train_size = int(n_rows*0.8)
	   
	X_train = X[0:train_size]
	y_train = y[0:train_size, :]
		
	X_test = X[train_size:n_rows]
	y_test = y[train_size:n_rows, :]
		  
	return (X_train, y_train, X_test, y_test)

def block35(net, scale=1.0, activation="relu"):
	tower_conv = relu(batch_normalization(conv_2d(net, 32, 1, bias=False, activation=None, name='Conv2d_1x1')))
	tower_conv1_0 = relu(batch_normalization(conv_2d(net, 32, 1, bias=False, activation=None,name='Conv2d_0a_1x1')))
	tower_conv1_1 = relu(batch_normalization(conv_2d(tower_conv1_0, 32, 3, bias=False, activation=None,name='Conv2d_0b_3x3')))
	tower_conv2_0 = relu(batch_normalization(conv_2d(net, 32, 1, bias=False, activation=None, name='Conv2d_0a_1x1')))
	tower_conv2_1 = relu(batch_normalization(conv_2d(tower_conv2_0, 48,3, bias=False, activation=None, name='Conv2d_0b_3x3')))
	tower_conv2_2 = relu(batch_normalization(conv_2d(tower_conv2_1, 64,3, bias=False, activation=None, name='Conv2d_0c_3x3')))
	tower_mixed = merge([tower_conv, tower_conv1_1, tower_conv2_2], mode='concat', axis=3)
	tower_out = relu(batch_normalization(conv_2d(tower_mixed, net.get_shape()[3], 1, bias=False, activation=None, name='Conv2d_1x1')))
	net += scale * tower_out
	if activation:
		if isinstance(activation, str):
			net = activations.get(activation)(net)
		elif hasattr(activation, '__call__'):
			net = activation(net)
		else:
			raise ValueError("Invalid Activation.")
	return net

def block17(net, scale=1.0, activation="relu"):
	tower_conv = relu(batch_normalization(conv_2d(net, 192, 1, bias=False, activation=None, name='Conv2d_1x1')))
	tower_conv_1_0 = relu(batch_normalization(conv_2d(net, 128, 1, bias=False, activation=None, name='Conv2d_0a_1x1')))
	tower_conv_1_1 = relu(batch_normalization(conv_2d(tower_conv_1_0, 160,[1,7], bias=False, activation=None, name='Conv2d_0b_1x7')))
	tower_conv_1_2 = relu(batch_normalization(conv_2d(tower_conv_1_1, 192, [7,1], bias=False, activation=None, name='Conv2d_0c_7x1')))
	tower_mixed = merge([tower_conv,tower_conv_1_2], mode='concat', axis=3)
	tower_out = relu(batch_normalization(conv_2d(tower_mixed, net.get_shape()[3], 1, bias=False, activation=None, name='Conv2d_1x1')))
	net += scale * tower_out
	if activation:
		if isinstance(activation, str):
			net = activations.get(activation)(net)
		elif hasattr(activation, '__call__'):
			net = activation(net)
		else:
			raise ValueError("Invalid Activation.")
	return net

def block8(net, scale=1.0, activation="relu"):
	tower_conv = relu(batch_normalization(conv_2d(net, 192, 1, bias=False, activation=None, name='Conv2d_1x1')))
	tower_conv1_0 = relu(batch_normalization(conv_2d(net, 192, 1, bias=False, activation=None, name='Conv2d_0a_1x1')))
	tower_conv1_1 = relu(batch_normalization(conv_2d(tower_conv1_0, 224, [1,3], bias=False, activation=None, name='Conv2d_0b_1x3')))
	tower_conv1_2 = relu(batch_normalization(conv_2d(tower_conv1_1, 256, [3,1], bias=False, name='Conv2d_0c_3x1')))
	tower_mixed = merge([tower_conv,tower_conv1_2], mode='concat', axis=3)
	tower_out = relu(batch_normalization(conv_2d(tower_mixed, net.get_shape()[3], 1, bias=False, activation=None, name='Conv2d_1x1')))
	net += scale * tower_out
	if activation:
		if isinstance(activation, str):
			net = activations.get(activation)(net)
		elif hasattr(activation, '__call__'):
			net = activation(net)
		else:
			raise ValueError("Invalid Activation.")
	return net

if __name__ == '__main__':
	generate_dataset_path('dataset.txt', True)

	X, Y = image_preloader(os.path.join(USER_ROOT, 'dataset.txt'), image_shape=(150, 150),
						   mode='file', categorical_labels=True, normalize=False)

	X, Y = shuffle(X, Y)
	X = 2 * (X / 255) - 1
	# X_train, y_train, X_test, y_test = split_training_testing(X, Y)
	# del X, Y

	num_classes = 3
	dropout_keep_prob = 0.8

	network = input_data(shape=[None, 150, 150, 3])
	conv1a_3_3 = relu(batch_normalization(conv_2d(network, 32, 3, strides=2, bias=False, padding='VALID', activation=None, name='Conv2d_1a_3x3')))
	del network
	conv2a_3_3 = relu(batch_normalization(conv_2d(conv1a_3_3, 32, 3, bias=False, padding='VALID', activation=None, name='Conv2d_2a_3x3')))
	del conv1a_3_3
	conv2b_3_3 = relu(batch_normalization(conv_2d(conv2a_3_3, 64, 3, bias=False, activation=None, name='Conv2d_2b_3x3')))
	del conv2a_3_3
	maxpool3a_3_3 = max_pool_2d(conv2b_3_3, 3, strides=2, padding='VALID', name='MaxPool_3a_3x3')
	del conv2b_3_3
	conv3b_1_1 = relu(batch_normalization(conv_2d(maxpool3a_3_3, 80, 1, bias=False, padding='VALID', activation=None, name='Conv2d_3b_1x1')))
	del maxpool3a_3_3
	conv4a_3_3 = relu(batch_normalization(conv_2d(conv3b_1_1, 192, 3, bias=False, padding='VALID', activation=None, name='Conv2d_4a_3x3')))
	del conv3b_1_1
	maxpool5a_3_3 = max_pool_2d(conv4a_3_3, 3, strides=2, padding='VALID', name='MaxPool_5a_3x3')
	del conv4a_3_3

	tower_conv = relu(batch_normalization(conv_2d(maxpool5a_3_3, 96, 1, bias=False, activation=None, name='Conv2d_5b_b0_1x1')))

	tower_conv1_0 = relu(batch_normalization(conv_2d(maxpool5a_3_3, 48, 1, bias=False, activation=None, name='Conv2d_5b_b1_0a_1x1')))
	tower_conv1_1 = relu(batch_normalization(conv_2d(tower_conv1_0, 64, 5, bias=False, activation=None, name='Conv2d_5b_b1_0b_5x5')))

	tower_conv2_0 = relu(batch_normalization(conv_2d(maxpool5a_3_3, 64, 1, bias=False, activation=None, name='Conv2d_5b_b2_0a_1x1')))
	tower_conv2_1 = relu(batch_normalization(conv_2d(tower_conv2_0, 96, 3, bias=False, activation=None, name='Conv2d_5b_b2_0b_3x3')))
	tower_conv2_2 = relu(batch_normalization(conv_2d(tower_conv2_1, 96, 3, bias=False, activation=None, name='Conv2d_5b_b2_0c_3x3')))

	tower_pool3_0 = avg_pool_2d(maxpool5a_3_3, 3, strides=1, padding='same', name='AvgPool_5b_b3_0a_3x3')
	tower_conv3_1 = relu(batch_normalization(conv_2d(tower_pool3_0, 64, 1, bias=False, activation=None, name='Conv2d_5b_b3_0b_1x1')))

	tower_5b_out = merge([tower_conv, tower_conv1_1, tower_conv2_2, tower_conv3_1], mode='concat', axis=3)

	net = repeat(tower_5b_out, 10, block35, scale=0.17)

	tower_conv = relu(batch_normalization(conv_2d(net, 384, 3, bias=False, strides=2,activation=None, padding='VALID', name='Conv2d_6a_b0_0a_3x3')))
	tower_conv1_0 = relu(batch_normalization(conv_2d(net, 256, 1, bias=False, activation=None, name='Conv2d_6a_b1_0a_1x1')))
	tower_conv1_1 = relu(batch_normalization(conv_2d(tower_conv1_0, 256, 3, bias=False, activation=None, name='Conv2d_6a_b1_0b_3x3')))
	tower_conv1_2 = relu(batch_normalization(conv_2d(tower_conv1_1, 384, 3, bias=False, strides=2, padding='VALID', activation=None, name='Conv2d_6a_b1_0c_3x3')))
	tower_pool = max_pool_2d(net, 3, strides=2, padding='VALID', name='MaxPool_1a_3x3')
	net = merge([tower_conv, tower_conv1_2, tower_pool], mode='concat', axis=3)
	net = repeat(net, 20, block17, scale=0.1)

	tower_conv = relu(batch_normalization(conv_2d(net, 256, 1, bias=False, activation=None, name='Conv2d_0a_1x1')))
	tower_conv0_1 = relu(batch_normalization(conv_2d(tower_conv, 384, 3, bias=False, strides=2, padding='VALID', activation=None, name='Conv2d_0a_1x1')))

	tower_conv1 = relu(batch_normalization(conv_2d(net, 256, 1, bias=False, padding='VALID', activation=None,name='Conv2d_0a_1x1')))
	tower_conv1_1 = relu(batch_normalization(conv_2d(tower_conv1, 288, 3, bias=False, strides=2, padding='VALID', activation=None, name='COnv2d_1a_3x3')))

	tower_conv2 = relu(batch_normalization(conv_2d(net, 256,1, bias=False, activation=None,name='Conv2d_0a_1x1')))
	tower_conv2_1 = relu(batch_normalization(conv_2d(tower_conv2, 288,3, bias=False, name='Conv2d_0b_3x3',activation=None)))
	tower_conv2_2 = relu(batch_normalization(conv_2d(tower_conv2_1, 320, 3, bias=False, strides=2, padding='VALID', activation=None, name='Conv2d_1a_3x3')))

	tower_pool = max_pool_2d(net, 3, strides=2, padding='VALID', name='MaxPool_1a_3x3')
	net = merge([tower_conv0_1, tower_conv1_1, tower_conv2_2, tower_pool], mode='concat', axis=3)

	net = repeat(net, 9, block8, scale=0.2)
	net = block8(net, activation=None)

	net = relu(batch_normalization(conv_2d(net, 1536, 1, bias=False, activation=None, name='Conv2d_7b_1x1')))
	net = avg_pool_2d(net, net.get_shape().as_list()[1:3], strides=2, padding='VALID', name='AvgPool_1a_8x8')
	net = flatten(net)
	net = dropout(net, dropout_keep_prob)
	loss = fully_connected(net, num_classes, activation='softmax')


	network = tflearn.regression(loss, optimizer='RMSprop',
						 loss='categorical_crossentropy', learning_rate=0.00005)
	model = tflearn.DNN(network, tensorboard_verbose=0)
	model.fit(X, Y, n_epoch=100, validation_set=0.1, shuffle=True,
			  show_metric=True, batch_size=32, snapshot_step=2000,
			  snapshot_epoch=False, run_id='inception_resnet_v2_cervical_cancer')

	model.save(os.path.join(USER_ROOT, 'cc_model'))
	# model.load(os.path.join(USER_ROOT, 'cc_model'))
	# print(model.evaluate(X_test, y_test))
	
	generate_dataset_path('test_dataset', False)
	for i in range(8):
		X_test, Y_test = image_preloader(os.path.join(USER_ROOT, 'test_dataset' + str(i) + '.txt'), image_shape=(150, 150),
							   mode='file', categorical_labels=True, normalize=False)
		X_test = np.array(X_test)
		X_test = 2 * (X_test / 255) - 1
		result = model.predict(X_test)
		df_data = pd.DataFrame(result)
		df_data.columns = TYPES[:3]
		img_names = []
		with open(os.path.join(USER_ROOT, 'test_dataset' + str(i) + '.txt')) as f:
			lines = f.readlines()
			for line in lines:
				name = line.split('\t')[0].split('/')[-1]
				img_names.append(name)
		df_name = pd.DataFrame(img_names)
		df_name.columns = ['image_name']
		df = pd.concat([df_name, df_data], axis=1)
		df.to_csv(os.path.join(USER_ROOT, 'submission.csv'), index=False, header=False, mode='a')
		del X_test, Y_test