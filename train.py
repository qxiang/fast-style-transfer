"""
Final Project - Style Transfer
CS1430 - Computer Vision
Brown University
"""

import time
import numpy as np
import tensorflow as tf
import PIL
import hyperparameters as hp
import run
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Activation, Add

class TransformNet(tf.keras.Model):
	def __init__(self):
		super(TransformNet, self).__init__()
		# Optimizer
		self.optimizer = tf.keras.optimizers.RMSprop(
			learning_rate=hp.learning_rate,
			momentum=hp.momentum)

		self.conv_step = [
			Conv2D(32, 9, 1, activation="relu", name="block1_conv1"),
			Conv2D(64, 3, 2, activation="relu", name="block1_conv2"),
			Conv2D(128, 3, 2, activation="relu", name="block1_conv3"),
		]

		self.res_block = [
			Conv2D(64, 3, 1, activation="relu", name="block2_conv1"),
			BatchNormalization(),
			Conv2D(64, 3, 2, activation=None, name="block1_conv2"),
			BatchNormalization(),
			Add(),
			Activation('relu')
		]

		self.deconv_step = [
			Conv2DTranspose(64, 3, 2, activation="relu", name="block3_deconv1"),
			Conv2DTranspose(32, 3, 2, activation="relu", name="block3_deconv2"),
			Conv2D(3, 9, 1, activation="relu", name="block3_conv1"),
		]
	
	def call(self, img):

		for layer in self.conv_step:
			img = layer(img)

		for i in range(5):
			for layer in self.res_block:
				img = layer(img)

		for layer in self.deconv_step:
			img = layer(img)
			
		return img

class StyleTransferModel(tf.keras.Model):
	def __init__(self):
		super(StyleTransferModel, self).__init__()
		self.vgg19 = run.VGGModel()
		self.vgg19.trainable = False
		# TODO: Initialize the real transform net
		self.transform = TransformNet()
	
	def call(self, img):
		return None

'''
	For convenience: I guess multi-style only
'''
@tf.function
def get_loss(feature_map):
	return None

def main():
	# TODO: Flowchart
	return None
