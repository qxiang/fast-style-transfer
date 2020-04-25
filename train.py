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

class TransformNet(tf.keras.Model):
	def __init__(self):
		super(TransformNet, self).__init__()
		# TODO
	
	def call(self, img):
		# TODO
		return None

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
