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
from run import preprocess, postprocess
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Activation

model = None
target_image = None
optimizer = tf.keras.optimizers.RMSprop(
				learning_rate=hp.learning_rate * 1e-4,
				momentum=hp.momentum)
target_path = "./trivial_result/output_content.jpg"

class ResBlock(tf.keras.Model):
	def __init__(self):
		super.__init__()
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
		super.__init__()
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
		content_layers_name = ['block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2', 'block3_conv1', 'block3_conv2']
		style_layers_name = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
		# Architecture
		self.vgg19 = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
		self.vgg19.trainable = False
		self.vgg19 = tf.keras.Model(
			inputs=[self.vgg19.input],
			outputs=[self.vgg19.get_layer(name).output for name in (content_layers_name + style_layers_name)])
		self.vgg19.trainable = False
	
	def call(self, img):
		""" Passes the image through the network. """
		result = self.vgg19(img)
		content, style = result[0:len(content_layers_name)], result[len(content_layers_name):len(result)]
		return np.array(content), np.array(style)


class StyleTransferModel(tf.keras.Model):
	def __init__(self, image, style):
		super().__init__()
		self.vgg19 = VGGModel()
		self.vgg19.trainable = False
		self.transform = TransformNet()
		self.transform.trainable = True
		self.image = image
		self.content_target, self.style_target = self.call(image)[0], self.call(style)[1]
	
	def call(self):
		return self.vgg19(self.transform(self.image))

	def get_transfer_img(self):
		return self.transform(self.image)

	def content_loss(self, content):
		# content_loss_val = tf.add_n([tf.reduce_sum((content[i] - content_target[i]) ** 2) / 2. for i in range(len(content))]) / len(content)
		return np.mean(np.square(content - self.content_target)) 

	def style_loss(self, style):
		# factor = [tf.cast((2 * tf.shape(style[i])[1] * tf.shape(style[i])[2]), tf.float32) ** 2 for i in range(len(style))]
		# style_loss_val = tf.add_n([tf.reduce_sum((style[i] - style_target[i]) ** 2) / factor[i] for i in range(len(style))]) / len(style)
		return np.mean(np.square(gram(style) - gram(self.style_target)))

	def get_loss(self):
		content, style = self.call(img)
		loss = hp.alpha * content_loss(content) + hp.beta * style_loss(content)
		return loss

def gram(style):
	style = np.tensordot(style, style, axes=(0, 1)),
	return style

@tf.function
def train_step(model):
	with tf.GradientTape() as g:
		loss = get_loss()

	train_vars = model.trainable_variables
	grad = g.gradient(loss, train_vars)
	optimizer.apply_gradients(zip(grad, train_vars))
	img = model.transfer_img()
	return loss, img

def main():
	content_image_path = './trivial_data/buildings.jpg'
	style_image_path = './trivial_data/starry-sky.jpg'
	content_image, style_image = preprocess(content_image_path), preprocess(style_image_path)
	model = StyleTransferModel(content_image, style_image)
	image = None
	for step in range(hp.num_step * hp.num_epoch):
		loss, image = train_step(model)
		tf.print(loss)

	# Output the image
	image = run.postprocess(image)
	image.save("./trivial_result/new_result.jpg")
	# test_image = preprocess("./trivial_data/architecture.jpg")
	# _, image = model(test_image)
	# image = postprocess(image)
	# image.save("./trivial_result/test_result.jpg")

if __name__ == "__main__":
	main()
