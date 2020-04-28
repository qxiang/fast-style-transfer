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
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Activation, Add, Concatenate

model = None
target_image = None
optimizer = tf.keras.optimizers.RMSprop(
				learning_rate=hp.learning_rate * 1e-4,
				momentum=hp.momentum)
target_path = "./trivial_result/output_content.jpg"

class ResBlock(tf.keras.Model):
	def __init__(self):
		super(ResBlock, self).__init__()
		self.model = [
			Conv2D(128, 3, 1, padding="same", activation="relu", name="block2_conv1"),
			BatchNormalization(),
			Conv2D(128, 3, 1, padding="same", activation=None, name="block2_conv2"),
			BatchNormalization()
		]
		self.concat = Concatenate()
		self.activation = Activation('relu')

	def call(self, img):
		residual = img
		for layer in self.model:
			img = layer(img)
		img = self.concat([img, residual])
		img = self.activation(img)
		return img

class TransformNet(tf.keras.Model):
	def __init__(self):
		super(TransformNet, self).__init__()
		self.conv_step = [
			Conv2D(32, 3, 1, padding="same", activation="relu", name="block1_conv1"),
			Conv2D(64, 3, 1, padding="same", activation="relu", name="block1_conv2"),
			Conv2D(128, 3, 1, padding="same", activation="relu", name="block1_conv3")
		]
		self.res_step = [ResBlock(), ResBlock(), ResBlock()]
		self.deconv_step = [
			Conv2D(64, 3, 1, padding="same", activation="relu", name="block3_conv1"),
			Conv2D(32, 3, 1, padding="same", activation="relu", name="block3_conv2"),
			#Conv2D(64, 3, 1, padding="same", activation="relu", name="block3_deconv1"),
			#Conv2D(32, 3, 1, padding="same", activation="relu", name="block3_deconv2"),
			Conv2D(16, 1, 1, padding="same", activation="relu", name="block3_conv3"),
			Conv2D(3, 1, 1, padding="same", activation=None, name="block3_conv4")
		]
	
	def call(self, img):
		for layer in self.conv_step:
			img = layer(img)
		for layer in self.res_step:
			img = layer(img)
		for layer in self.deconv_step:
			img = layer(img)
		return img

class StyleTransferModel(tf.keras.Model):
	def __init__(self):
		super(StyleTransferModel, self).__init__()
		self.vgg19 = run.VGGModel()
		self.vgg19.trainable = False
		self.transform = TransformNet()
		self.transform.trainable = True
	
	def call(self, img):
		img = self.transform(img)
		feature_map = self.vgg19(img)
		return feature_map, img

'''
For convenience: I guess we need to change it for multi-style
'''
@tf.function
def get_loss(feature_map, img):
	#5e+6
	loss = tf.reduce_sum((target_image - img) ** 2)
	#loss += run.get_loss(feature_map)
	return loss

@tf.function
def train_step(image):
	global model
	with tf.GradientTape() as g:
		feature_map, img = model(image)
		loss = get_loss(feature_map, img)
	train_vars = model.trainable_variables
	grad = g.gradient(loss, train_vars)
	optimizer.apply_gradients(zip(grad, train_vars))
	return loss, img

def main():
	global model, target_image
	model = StyleTransferModel()
	# Generate data for content
	#data_gen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg19.preprocess_input)
	#data_gen = data_gen.flow_from_directory("./train_data", target_size=(hp.img_size, hp.img_size), class_mode=None, batch_size=1)
	# TODO: Train the net
	# Import the images
	run.content_image, run.style_image = run.preprocess(run.content_image_path), run.preprocess(run.style_image_path)
	target_image = run.preprocess(target_path)
	# Create model
	run.model = run.VGGModel()
	# Extract content and style
	run.content_target, _ = run.model(run.content_image)
	_, run.style_target = run.model(run.style_image)
	run.style_target = [run.gram(style) for style in run.style_target]
	image = None
	for step in range(hp.num_step * hp.num_epoch):
		loss, image = train_step(run.content_image)
		tf.print(loss)
	model.trainable = False
	# Output the image
	image = run.postprocess(image)
	image.save("./trivial_result/new_result.jpg")
	test_image = run.preprocess("./trivial_data/architecture.jpg")
	_, image = model(test_image)
	image = run.postprocess(image)
	image.save("./trivial_result/test_result.jpg")

if __name__ == "__main__":
	main()
