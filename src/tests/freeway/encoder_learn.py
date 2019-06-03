from comet_ml import Experiment
experiment = Experiment(api_key="8gFIuv61aMnLn2YmtGHULdr1P", project_name="encoder_learn", workspace="basarane")

import argparse

parser = argparse.ArgumentParser(description='Encoder Training')
parser.add_argument('output_path', type=str, default='test_output_r3', help='Output path')

args = parser.parse_args()

import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Lambda, Concatenate, Add, Activation, BatchNormalization, LeakyReLU, ReLU, PReLU, ELU
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *

import tensorflow.keras.backend as K
from tensorflow.keras.backend import set_session
import random 

import os
import sys
from pathlib import Path

np.set_printoptions(threshold=sys.maxsize)

#os.environ['CUDA_VISIBLE_DEVICES'] = ''
output_path = args.output_path + "/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

libroot = Path(os.path.realpath(__file__)).parent.parent.parent 
print('libroot', libroot)
sys.path.append(str(libroot))

from utils.summary_writer import SummaryWriter
summary_writer = tf.summary.FileWriter(output_path, K.get_session().graph) 
sw = SummaryWriter(summary_writer, ['Loss'])

config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.per_process_gpu_memory_fraction = 0.1
config.gpu_options.allow_growth = True

set_session(tf.Session(config=config))

cc = 13 # car count
input_shape_orig = (210,160, cc)
offsety = 24
sub_range = ((offsety,offsety+cc*20), (20,100))
sub_range = ((0,input_shape_orig[0]),(0,input_shape_orig[1]))
sub_range = ((0,20),(0,20))
input_shape = (sub_range[0][1] - sub_range[0][0], sub_range[1][1] - sub_range[1][0], cc)

use_batch_norm = False


from enum import Enum
class TrainNetwork(Enum):
	CHANNELED_AUTO_ENCODER = 1
	ENCODER = 2
	
#(32,4),
# Nadam defaults
# lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004
train_network = TrainNetwork.CHANNELED_AUTO_ENCODER
activation = "relu"
#activation = "PReLU"
#activation = "LeakyReLU"
optimizer = Nadam(lr=0.0001, )
#optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
conv_info = [(256,3, True, False),(128,3, True, False),(64,3, False, False),(64,3, False, False),(32,3, False, False),(32,3, False, False),(32,3, False, False),(32,3, False, False),(32,3, False, False),(1,3, False, False), (32,3,False,True), (cc,3, False, True)]

# @derya
#conv_info = [(32,4),(32,4),(32,4),(32,4),(32,4),(32,3)]
# 32x64x128 artan filtre sayisi ve azalan filtre sayisi dene
# 5 veya 5'den buyuk filtreleri kullanmamaya calis
# activation degistirmeyi dene, leaky relu dene. sigmoid ve tanh dene (tanh'in daha iyi oldugu bilgisi geldi)
# ara layer'lara dropout koy (onunla birlikte l2 dene)
	
	
def input_slice_function(I):
	def input_slice(x):
		return x[:,:,:,I:I+1]
	return input_slice

def get_decoder(input_shape):
	cc = input_shape[2]
	input = Input(shape=input_shape, name='encoded_observation')
	conv_layer_defs = []
	layers = []
	for I in range(cc):
		conv_layer_def = Conv2DTranspose(filters=4, kernel_size=10, strides=1, activation='relu', padding='same',kernel_initializer='glorot_normal')
		conv_layer_defs.append(conv_layer_def)
		sliced_input = Lambda(input_slice_function(I))(input)
		layer = conv_layer_def(sliced_input)
		layers.append(layer)
	result =  Concatenate(axis=3)(layers)
	model_decoder = Model(inputs = [input], outputs = [result], name="decoder")
	return model_decoder, conv_layer_defs

def set_decoder_weights(conv_layer_defs):
	for I,conv_layer_def in enumerate(conv_layer_defs):
		weights = conv_layer_def.get_weights()
		new_weights = np.zeros((10,10,4,1))
		im = Image.open(f'images/trans/car{I+1}.png')
		imarr = np.array(im)
		print(imarr.shape, f'{(4*I)}:{(4*(I+1))}')
		new_weights[:,:,:,0] = imarr / 255.0 #np.expand_dims(imarr, 3)
		weights[0] = new_weights
		conv_layer_def.set_weights(weights)

def get_encoder(input_shape, conv_info):
	cc = input_shape[2]
	input = Input(shape=(input_shape[0],input_shape[1],4), name='observation')
	convs = []
	conv_layer_defs = []
	batchnorm_layer_defs = []
	activation_defs = []
	for I in range(cc):
		defs = []
		batchnorms = []
		activations = []
		encoded = input
		for depth,info in enumerate(conv_info,1):
			if info[3]:
				continue
			if info[2] and I>0:
				conv = conv_layer_defs[0][depth-1]
			else:
				conv = Conv2D(filters=info[0],kernel_size=info[1],strides=1,padding="same",kernel_initializer='glorot_normal') #RandomNormal(mean=0.0, stddev=0.001, seed=None)
			batchnorm = BatchNormalization()
			encoded = conv(encoded)
			if use_batch_norm:
				encoded = batchnorm(encoded)
			if depth == len(conv_info):
				activation_fn = ReLU()
			else:
				if activation == "PReLU":
					activation_fn = PReLU(shared_axes=[1, 2])
				elif activation == "LeakyReLU":
					activation_fn = LeakyReLU()
				else:
					activation_fn = Activation(activation)
			encoded = activation_fn(encoded)
			defs.append(conv)
			batchnorms.append(batchnorm)
			activations.append(activation_fn)
		convs.append(encoded)
		conv_layer_defs.append(defs)
		batchnorm_layer_defs.append(batchnorms)
		activation_defs.append(activations)
	result = Concatenate(axis=3)(convs)
	defs = []
	activations = []
	for depth,info in enumerate(conv_info,1):
		if info[3]:
			conv = Conv2D(filters=info[0],kernel_size=info[1],strides=1,padding="same",kernel_initializer='glorot_normal') #RandomNormal(mean=0.0, stddev=0.001, seed=None)
			result = conv(result)
			if activation == "PReLU":
				activation_fn = PReLU(shared_axes=[1, 2])
			elif activation == "LeakyReLU":
				activation_fn = LeakyReLU()
			else:
				activation_fn = Activation(activation)
			result = activation_fn(result)
			defs.append(conv)
			activations.append(activation_fn)
	conv_layer_defs.append(defs)
	activation_defs.append(activations)
	model_encoder = Model(inputs = [input], outputs = [result], name="encoder")
	return model_encoder, conv_layer_defs, batchnorm_layer_defs, activation_defs
	
def merge_tensor(x):
	# @ersin - bu simple bir algoritma, gerekirse over operator'uyle degistir https://en.wikipedia.org/wiki/Alpha_compositing
	output_color = x[0][:,:,:,0:3]
	output_alpha = x[0][:,:,:,3:]
	for layer in x[1:]:
		output_alpha_new = output_alpha + layer[:,:,:,3:] * (1-output_alpha)
		output_alpha_new_fixed = K.switch(K.equal(output_alpha_new,0), output_alpha_new+1 ,output_alpha_new)
		output_color = (output_color * output_alpha + layer[:,:,:,0:3] * layer[:,:,:,3:] * (1-output_alpha))/output_alpha_new_fixed
		output_alpha = output_alpha_new
		#output[:,:,:,0:3] = (output[:,:,:,0:3] * output[:,:,:,3:] + layer[:,:,:,0:3] * layer[:,:,:,3:] * (1-output[:,:,:,3:]))/(output[:,:,:,3:]+layer[:,:,:,3:]*(1-output[:,:,:,3:]))
		#output[:,:,:,3:] = output[:,:,:,3:] + layer[:,:,:,3:] * (1-output[:,:,:,3:])
		#mask = output[:,:,:,3:]>0.0001
		#mask = K.repeat_elements(mask, rep=4, axis=3)
		#output = K.switch(mask,output,layer)
	return K.concatenate([output_color,output_alpha], axis=3)
	
def get_large_decoder(input_shape_orig, deconv_layer_defs, seperate_channels = False):
	input = Input(shape=input_shape_orig, name='encoded_observation_large')
	layers = []
	for I in range(cc):
		sliced_input = Lambda(input_slice_function(I))(input)
		layer = deconv_layer_defs[I](sliced_input)
		layers.append(layer)
	if seperate_channels:
		result = Concatenate(axis=3)(layers)
	else:
		#result = Add()(layers)
		result = Lambda(merge_tensor)(layers)
	model_decoder = Model(inputs = [input], outputs = [result], name="decoder_large")
	return model_decoder

def get_large_encoder(input_shape_orig, conv_layer_defs, batchnorm_layer_defs, activation_layer_defs):
	cc = input_shape_orig[2]
	input = Input(shape=(input_shape_orig[0],input_shape_orig[1],4), name='observation_large')
	convs = []
	for I in range(cc):
		encoded = input
		for idx,conv in enumerate(conv_layer_defs[I]):
			encoded = conv(encoded)
			if use_batch_norm:
				encoded = batchnorm_layer_defs[I][idx](encoded)
			encoded = activation_layer_defs[I][idx](encoded)
		convs.append(encoded)
	result =  Concatenate(axis=3)(convs)
	for depth,(conv,activation) in enumerate(zip(conv_layer_defs[-1],activation_layer_defs[-1]),1):
		result = conv(result)
		result = activation(result)
	model_encoder = Model(inputs = [input], outputs = [result], name="encoder_large")
	return model_encoder

model_decoder, deconv_layer_defs = get_decoder(input_shape)
set_decoder_weights(deconv_layer_defs)
model_encoder, conv_layer_defs, batchnorm_layer_defs, activation_defs = get_encoder(input_shape, conv_info)

observations = Input(shape=(input_shape[0],input_shape[1], 4), name="observation2")
repr = model_encoder(observations)
for layer in model_decoder.layers:
    layer.trainable = False

predicted = model_decoder(repr)
model = Model(inputs=[observations], outputs=predicted)
if train_network == TrainNetwork.CHANNELED_AUTO_ENCODER:
	model.compile(optimizer=optimizer,loss='mse')
elif train_network == TrainNetwork.ENCODER:
	model_encoder.compile(optimizer=optimizer,loss='mse')

# @derya
# default degerlerin disina cik, momentuma ozellikle bak, dusurmeyi dene
# diger optimizer'lara da tekrar bakabilirsin

model_large_decoder = get_large_decoder(input_shape_orig, deconv_layer_defs)
model_large_decoder_channels = get_large_decoder(input_shape_orig, deconv_layer_defs, True)
model_large_encoder = get_large_encoder(input_shape_orig, conv_layer_defs, batchnorm_layer_defs, activation_defs)
observations_large = Input(shape=(input_shape_orig[0],input_shape_orig[1], 4), name="observation2_large")
repr_large = model_large_encoder(observations_large)
predicted_large = model_large_decoder(repr_large)
model_large = Model(inputs=[observations_large], outputs=[predicted_large])

def save_image(outmap, batch_size, fname):
	outmap[outmap>1] = 1
	outmap[outmap<0] = 0
	if len(outmap.shape) == 4:
		output_image = outmap[:,:,:,:] * 255.0
	else:
		output_image = outmap[:,:,:] * 255.0
	output_image = output_image.astype(np.uint8)
	for J in range(batch_size):
		if len(outmap.shape) == 4:
			output_im = Image.fromarray(output_image[J,:,:,:])
		else:
			output_im = Image.fromarray(output_image[J,:,:])
		output_im.save(fname + f'_{J}.png')
def save_encoded(outmap, batch_size, fname):
	output_image = outmap[:,:,:,:] * 255.0
	output_image = output_image.astype(np.uint8)
	for J in range(batch_size):
		for K in range(output_image.shape[3]):
			output_im = Image.fromarray(output_image[J,:,:,K])
			output_im.save(fname + f'_{J}_{K}.png')		

def merge_image(im1, im2):
	im = im1.copy()
	empty = im[:,:,3] < 0.0001
	im[empty] = im2[empty]
	return im
	
batch_size = cc*10
for I in range(20000):
	inmap = np.zeros((batch_size,) + input_shape)
	genmap = np.zeros((batch_size,) + (input_shape[0],input_shape[1],4))
	Ks = []
	# @ersin random yapma, her batch'de her sample'dan esit adet olsun
	for J in range(0,batch_size):
		#K = random.randint(0, cc-1)
		K = J%cc
		K2 = -1
		if np.random.random()<=0.5:
			K2 = cc - 3 + np.random.randint(0,3)
		Ks.append([K, K2])
		r1 = random.randint(0, input_shape[1]-1)
		r2 = random.randint(5, input_shape[0]-6)
		r3 = random.randint(5, input_shape[1]-6)
		r4 = random.randint(5, input_shape[0]-6)
		inmap[J,r2,r1,K] = 1
		if K2>=0 and K<10:
			inmap[J,r4,r3,K2] = 1
	gen_image = model_decoder.predict_on_batch([inmap])
	for J in range(0,batch_size):
		if Ks[J][1]>=0:
			genmap[J,:,:,:] = merge_image(gen_image[J,:,:,Ks[J][0]*4:Ks[J][0]*4+4],gen_image[J,:,:,Ks[J][1]*4:Ks[J][1]*4+4])
		else:
			genmap[J,:,:,:] = gen_image[J,:,:,Ks[J][0]*4:Ks[J][0]*4+4]
	if train_network == TrainNetwork.CHANNELED_AUTO_ENCODER:
		res = model.train_on_batch([genmap], gen_image) 
	elif train_network == TrainNetwork.ENCODER:
		res = model_encoder.train_on_batch([genmap], [inmap]) 
	sw.add([res], I)

	if I%100 == 0:
	#	save_image(gen_image[:1,:,:,Ks[0]*3:Ks[0]*3+3], 1, f'{output_path}car_generated_{I}')
		#save_image(genmap[:,:,:,:], batch_size, f'{output_path}train_sample_{I}')
		save_image(genmap[:1,:,:,:], 1, f'{output_path}train_sample_{I}')
		if True:
			model_large_decoder.save(f'{output_path}{I}_model_large_decoder.h5')
			model_large_encoder.save(f'{output_path}{I}_model_large_encoder.h5')
			model_large_decoder_channels.save(f'{output_path}{I}_model_large_decoder_channels.h5')
			model_large.save(f'{output_path}{I}_model_large.h5')
			model.save(f'{output_path}{I}_model.h5')

			inmap = np.zeros((1,) + input_shape_orig)
			for K in range(cc):
				r1 = random.randint(0, input_shape_orig[1]-1)
				r2 = random.randint(0, input_shape_orig[0]-1)
				if K<cc-3:
					inmap[0,24 + 11 + 15*K,r1,K] = 1
				else:
					inmap[0,r2,r1,K] = 1
			gen_image = model_large_decoder.predict_on_batch([inmap])
			save_image(gen_image, 1, f'{output_path}{I}_gen')
			learned_image = model_large.predict_on_batch([gen_image[0:1,:,:,:]])
			learned_encode = model_large_encoder.predict_on_batch([gen_image[0:1,:,:,:]])
			save_image(learned_image, 1, f'{output_path}{I}_learned')
			for K in range(cc):
				save_image(learned_encode[:,:,:,K], 1, f'{output_path}{I}_encoded_{K}')
	print(f"{I}\t{res}")
	