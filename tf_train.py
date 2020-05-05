"""
Final Project - Style Transfer
CS1430 - Computer Vision
Brown University
"""

import time
import numpy as np
import tensorflow as tf
import hyperparameters as hp
from run import preprocess, postprocess
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Activation
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

model = None
target_image = None
optimizer = tf.keras.optimizers.Adam(
				learning_rate=hp.learning_rate * 1e-4,
				momentum=hp.momentum)
target_path = "./trivial_result/output_content.jpg"

class ResBlock(tf.keras.Model):
	def __init__(self):
		super().__init__()
		self.model = [
			Conv2D(128, 3, 1, padding="same", name="block2_conv1"),
			BatchNormalization(),
			Activation('relu'),
			Conv2D(128, 3, 1, padding="same", name="block2_conv2"),
			BatchNormalization()
		]
		self.activation = Activation('relu')

	def call(self, img):
		residual = img
		for layer in self.model:
			img = layer(img)
		return self.activation(img + residual)

class TransformNet(tf.keras.Model):
	def __init__(self):
		super().__init__()
		self.conv_step = [
			Conv2D(32, 3, 2, padding="same", name="block1_conv1"),
			BatchNormalization(),
			Activation('relu'),
			Conv2D(64, 3, 2, padding="same", name="block1_conv2"),
			BatchNormalization(),
			Activation('relu'),
			Conv2D(128, 3, 2, padding="same", name="block1_conv3"),
			BatchNormalization(),
			Activation('relu')
		]
		self.res_step = [ResBlock(), ResBlock(), ResBlock(), ResBlock(), ResBlock()]
		self.deconv_step = [
			Conv2DTranspose(64, 3, 2, padding="same", name="block3_conv1"),
			BatchNormalization(),
			Activation('relu'),
			Conv2DTranspose(32, 3, 2, padding="same", name="block3_conv2"),
			BatchNormalization(),
			Activation('relu'),
			Conv2DTranspose(3, 1, 2, padding="same", name="block3_conv3"),
			BatchNormalization(),
			Activation('tanh')
		]
		self.model = self.conv_step + self.res_step + self.deconv_step
	
	def call(self, img):
		for layer in self.model:
			img = layer(img)
		return img

class VGGModel(tf.keras.Model):
	def __init__(self):
		super().__init__()
		self.content_layers_name = ['block2_conv2']
		
		self.style_layers_name = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3'] 
		# Architecture 
		self.vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet') 
		self.vgg = tf.keras.Model(inputs=[self.vgg.input], 
			outputs=[self.vgg.get_layer(name).output for name in (self.content_layers_name + self.style_layers_name)])
		self.vgg.trainable = False 
	
	def call(self, img):
		""" Passes the image through the network. """
		result = self.vgg(img)
		content, style = result[0], result[1:len(result)]
		style = [x[0] for x in style]
		return content, style

class StyleTransferModel(tf.keras.Model):
	def __init__(self, image, style):
		super().__init__()
		self.vgg = VGGModel()
		self.transform = TransformNet()
		self.image = image
		self.content_target, self.style_target = self.call(image)[0], self.call(style)[1]
	
	def call(self, image):
		return self.vgg(self.transform(image))

	def get_transfer_img(self):
		return self.transform(self.image)

	def content_loss(self, content):
		loss = 0
		num = 0
		for feat, feat_target in zip(content, self.content_target):
			loss += tf.reduce_sum(tf.square(feat - feat_target))
			num += tf.size(feat, out_type=tf.dtypes.float32)
		return loss / num

	def style_loss(self, style):
		loss = 0
		num = 0
		for feat, feat_target in zip(style, self.style_target):
			loss += tf.reduce_sum(tf.square(gram(feat) - gram(feat_target)))
			num += tf.size(feat, out_type=tf.dtypes.float32)
		return loss / num

	def get_loss(self):
		content, style = self.call(self.image)
		loss = hp.alpha * self.content_loss(content) + hp.beta * self.style_loss(style)
		return loss

def gram(style):
	return tf.tensordot(style, style, axes=(0, 1)) 

def train_step(model):
	with tf.GradientTape() as g:
		loss = model.get_loss()

	train_vars = model.trainable_variables
	grad = g.gradient(loss, train_vars)
	optimizer.apply_gradients(zip(grad, train_vars))
	return loss 

def main():
	content_image_path = './trivial_data/buildings.jpg'
	style_image_path = './trivial_data/starry-sky.jpg'
	content_image, style_image = preprocess(content_image_path), preprocess(style_image_path)
	model = StyleTransferModel(content_image, style_image)
	for step in range(5):
		loss = train_step(model)
		tf.print(loss)
	image = model.get_transfer_img()	
	image = postprocess(image)
	image.save("./trivial_result/new_result.jpg")

if __name__ == "__main__":
	main()
