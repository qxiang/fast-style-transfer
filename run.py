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
from tensorflow.keras.preprocessing.image import load_img, img_to_array

content_image_path = './trivial_data/buildings.jpg'
style_image_path = './trivial_data/starry-sky.jpg'
generate_image_path = ['./trivial_result/output_noise', './trivial_result/output_content']
generate_image_path_ext = '.jpg'

content_layers_name = ['block2_conv2', 'block4_conv2']
style_layers_name = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

optimizer = tf.keras.optimizers.RMSprop(
		learning_rate=hp.learning_rate,
		momentum=hp.momentum)

content_target, style_target = None, None
model = None

def preprocess(image_path):
	img = load_img(image_path, target_size=(hp.img_size, hp.img_size), interpolation='bicubic')
	img = img_to_array(img)
	img = np.expand_dims(img, axis=0)
	img = tf.keras.applications.vgg19.preprocess_input(img)
	return img

class VGGModel(tf.keras.Model):
	def __init__(self):
		super(VGGModel, self).__init__()
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
		return content, style

def content_loss(content):
	content_loss_val = tf.add_n([tf.reduce_sum((content[i] - content_target[i]) ** 2) / 2. for i in range(len(content))]) / len(content)
	return content_loss_val

def style_loss(style):
	factor = [tf.cast((2 * tf.shape(style[i])[1] * tf.shape(style[i])[2]), tf.float32) ** 2 for i in range(len(style))]
	style = [gram(map) for map in style]
	style_loss_val = tf.add_n([tf.reduce_sum((style[i] - style_target[i]) ** 2) / factor[i] for i in range(len(style))]) / len(style)
	return style_loss_val

def gram(style):
	style = tf.tensordot(style, style, axes=[[1, 2], [1, 2]])
	style = tf.squeeze(style, axis=2)
	return style

@tf.function
def get_loss(feature_map):
	content, style = feature_map
	content_loss_val = content_loss(content)
	style_loss_val = style_loss(style)
	total_loss = style_loss_val * hp.beta + content_loss_val * hp.alpha
	return total_loss

@tf.function
def train_step(image):
	with tf.GradientTape() as g:
		g.watch(image)
		feature_map = model(image)
		loss = get_loss(feature_map)
	grad = g.gradient(loss, image)
	optimizer.apply_gradients([(grad, image)])
	return loss

def postprocess(tensor):
	tensor = tf.reshape(tensor, (hp.img_size, hp.img_size, 3))
	tensor = np.array(tensor, dtype=np.float32)
	# Cheat mean of ImageNet database from Internet
	mean = [103.939, 116.779, 123.68]
	tensor += mean
	tensor = np.clip(tensor, 0., 255.)
	result = np.copy(tensor)
	result[:,:,2] = tensor[:,:,0]
	result[:,:,0] = tensor[:,:,2]
	return PIL.Image.fromarray(result.astype(np.uint8))

def main(useNoise=False):
	global content_target, style_target, model
	# Import the images
	content_image, style_image = preprocess(content_image_path), preprocess(style_image_path)
	# Create model
	model = VGGModel()
	# Extract content and style
	content_target, _ = model(content_image)
	_, style_target = model(style_image)
	style_target = [gram(style) for style in style_target]
	# Initialize generated image and optimizer
	if useNoise:
		generate_image = tf.Variable(tf.random.uniform(shape=content_image.shape, minval=-1., maxval=1.))
	else:
		generate_image = tf.Variable(content_image)
	# Pick a very large loss placeholder
	loss = tf.float32.max
	# Start transfer
	total_time = 0.
	image = None
	for epoch in range(hp.num_epoch):
		current_loss = 0.
		start = time.time()
		for step in range(hp.num_step):
			current_loss = train_step(generate_image)
		total_time += time.time() - start
		# Output the image
		if current_loss < loss:
			loss = current_loss
			image = postprocess(generate_image)
			if useNoise:
				image.save(generate_image_path[0] + '_' + str(epoch) + generate_image_path_ext)
			else:
				image.save(generate_image_path[1] + '_' + str(epoch) + generate_image_path_ext)
	# Output final image
	print(total_time)
	if useNoise:
		image.save(generate_image_path[0] + generate_image_path_ext)
	else:
		image.save(generate_image_path[1] + generate_image_path_ext)

if __name__ == "__main__":
	main()
	main(True)
