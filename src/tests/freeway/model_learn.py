from comet_ml import Experiment
experiment = Experiment(api_key="8gFIuv61aMnLn2YmtGHULdr1P", project_name="model_learn", workspace="basarane")

import argparse

parser = argparse.ArgumentParser(description='Encoder Training')
parser.add_argument('encoder_path', type=str, default=None, help='Encoder path')
parser.add_argument('encoder_idx', type=int, default=None, help='Encoder idx')
parser.add_argument('output_path', type=str, default=None, help='Output path')

args = parser.parse_args()

import gym

env = gym.make('FreewayDeterministic-v4')

import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Lambda, Concatenate, Add, BatchNormalization, Reshape, ZeroPadding2D, Cropping2D, Conv1D
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.regularizers import l1,l2,l1_l2
from tensorflow.keras.utils import to_categorical

import tensorflow.keras.backend as K
from tensorflow.keras.backend import set_session
import random 

import os
import sys

seed = 100
np.random.seed(seed)
random.seed(seed)
env.seed(seed)
env.action_space.np_random.seed(123)
tf.set_random_seed(seed)

np.set_printoptions(threshold=sys.maxsize)

#os.environ['CUDA_VISIBLE_DEVICES'] = ''
#autoencoder_dir = 'test_output_e1/'
#autoencoder_idx = 1900
#autoencoder_dir = 'test_output_e2/'
#autoencoder_idx = 1900
autoencoder_dir = args.encoder_path
autoencoder_idx = args.encoder_idx

import sys
import sys
from pathlib import Path

output_path = None
if not args.output_path is None:
    output_path = args.output_path + "/"
    #output_path = "test_output_m1/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
libroot = Path(os.path.realpath(__file__)).parent.parent.parent  #
print('libroot', libroot)
sys.path.append(str(libroot))

from utils.summary_writer import SummaryWriter
from utils.preprocess import remove_bg

sw = None
if not output_path is None:
    summary_writer = tf.summary.FileWriter(output_path, K.get_session().graph) 
    sw = SummaryWriter(summary_writer, ['Loss'])
    
config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.per_process_gpu_memory_fraction = 0.1
config.gpu_options.allow_growth = True

set_session(tf.Session(config=config))

cc = 13 # car count
input_shape_orig = (210,160, cc)
sub_range = ((0,20),(0,20))
input_shape = (sub_range[0][1] - sub_range[0][0], sub_range[1][1] - sub_range[1][0], cc)

def action_merge(x):
    layers = [x[0][:,:,:,:]]
    for I in range(x[1].shape[1]):
        a =  K.expand_dims(x[1][:,I:I+1], axis=-1) 
        a =  K.expand_dims(a, axis=-1) 
        layers.append(x[0][:,:,:,:]*a)
    return K.concatenate(layers, axis=3)

def channel_slice_function(I,J):
    def input_slice(x):
        return x[:,:,:,I:J]
    return input_slice

def image_slice_function(s):
    def input_slice(x):
        return x[:,s[0][0]:s[0][1],s[1][0]:s[1][1],:]
    return input_slice

def get_decoder(input_shape):
    cc = input_shape[2]
    input = Input(shape=input_shape, name='encoded_observation')
    conv_layer_defs = []
    layers = []
    for I in range(cc):
        conv_layer_def = Conv2DTranspose(filters=4, kernel_size=10, strides=1, activation='relu', padding='same',kernel_initializer='glorot_normal')
        conv_layer_defs.append(conv_layer_def)
        sliced_input = Lambda(channel_slice_function(I,I+1))(input)
        layer = conv_layer_def(sliced_input)
        layers.append(layer)
    result =  Concatenate(axis=3)(layers)
    model_decoder = Model(inputs = [input], outputs = [result], name="decoder")
    return model_decoder, conv_layer_defs

def merge_tensor(x):
    # @ersin - bu simple bir algoritma, gerekirse over operator'uyle degistir https://en.wikipedia.org/wiki/Alpha_compositing
    output_color = x[0][:,:,:,0:3]
    output_alpha = x[0][:,:,:,3:]
    for layer in x[1:]:
        output_alpha_new = output_alpha + layer[:,:,:,3:] * (1-output_alpha)
        output_alpha_new_fixed = K.switch(K.equal(output_alpha_new,0), output_alpha_new+0.001 ,output_alpha_new)
        output_color = (output_color * output_alpha + layer[:,:,:,0:3] * layer[:,:,:,3:] * (1-output_alpha))/output_alpha_new_fixed
        output_alpha = output_alpha_new
    return K.concatenate([output_color,output_alpha], axis=3)
    
def get_large_decoder(input_shape_orig, deconv_layer_defs, seperate_channels = False):
    input = Input(shape=input_shape_orig, name='encoded_observation_large')
    layers = []
    for I in range(cc):
        sliced_input = Lambda(channel_slice_function(I,I+1))(input)
        layer = deconv_layer_defs[I](sliced_input)
        layers.append(layer)
    if seperate_channels:
        result = Concatenate(axis=3)(layers)
    else:
        #result = Add()(layers)
        result = Lambda(merge_tensor)(layers)
    model_decoder = Model(inputs = [input], outputs = [result], name="decoder_large")
    return model_decoder

model_ae_decoder = tf.keras.models.load_model(f'{autoencoder_dir}{autoencoder_idx}_model_large_decoder.h5')
model_ae_decoder_channels = tf.keras.models.load_model(f'{autoencoder_dir}{autoencoder_idx}_model_large_decoder_channels.h5')
model_ae_encoder = tf.keras.models.load_model(f'{autoencoder_dir}{autoencoder_idx}_model_large_encoder.h5')
#model_ae = tf.keras.models.load_model(f'{autoencoder_dir}{autoencoder_idx}_model_large.h5')

model_decoder, deconv_layer_defs = get_decoder(input_shape)

model_large_decoder = get_large_decoder(input_shape_orig, deconv_layer_defs)
model_large_decoder.set_weights(model_ae_decoder.get_weights())
model_ae_decoder = model_large_decoder

#model_large_decoder_channels = get_large_decoder(input_shape_orig, deconv_layer_defs, True)
#model_large_decoder_channels.set_weights(model_ae_decoder_channels.get_weights())
#model_ae_decoder_channels = model_large_decoder_channels

cross_timer_count = 7
carpisma_timer_count = 13

def sum_tavuk(x):
    return K.sum(x,axis=[1,2,3],keepdims=True)

def add_to_timer(timer_count):
    def add_to_timer_fn(x):
        tavuk_crossed = x[1]
        if len(tavuk_crossed.shape)>2:
            tavuk_crossed = K.sum(tavuk_crossed,axis=[1,2,3],keepdims=False)
        tavuk_crossed = K.expand_dims(tavuk_crossed, axis=1)
        tavuk_crossed = K.expand_dims(tavuk_crossed, axis=1)
        not_crossed = 1-K.repeat_elements(tavuk_crossed,timer_count,axis=1)
        tavuk_crossed = K.temporal_padding(tavuk_crossed, (0,timer_count-1))
        res = tavuk_crossed + x[0]*not_crossed
        return res
    return add_to_timer_fn

def is_crossed_fn(x):
    return K.sum(x,axis=[1,2],keepdims=False)

# action x[0]: (?,3), is_crossed x[1]: (?,1), output: (?,3) eger is_crossed>0.1 ise [1,0,0] donuyor, yoksa [1,1,1]
# yapilacak degisiklik, artik x[2] de gelecek (carpisma_timer_input)
# output (?,3+carpisma_timer_count) olacak
# eger carpisma_timer_input'un toplami 0.1'den buyukse ilk 3'u sifirlanacak, carpisma_timer_input concatenate edilecek
def merge_action_crossed_fn(x):
    input = x[0]
    is_crossed_orig = x[1]
    is_crossed_orig = K.switch(is_crossed_orig>0.1,K.maximum(is_crossed_orig,1),is_crossed_orig*0)
    action_count = input.shape[1]
    is_crossed = K.expand_dims(is_crossed_orig, axis=1)
    is_crossed = K.expand_dims(is_crossed, axis=1)
    is_crossed = K.temporal_padding(is_crossed, (0,action_count-1))
    is_crossed = K.squeeze(is_crossed,axis=2)
    is_crossed_mask = K.expand_dims(is_crossed_orig, axis=1)
    is_crossed_mask = K.repeat_elements(is_crossed_mask,action_count,axis=1)
    res_crossed = (1-is_crossed_mask) * input + is_crossed
    carpisma_timer_orig = x[2]
    carpisma_timer_orig = K.squeeze(carpisma_timer_orig, axis=2)
    is_carpisma = K.sum(carpisma_timer_orig,axis=1)
    is_carpisma = K.switch(is_carpisma>0.1,K.maximum(is_carpisma,1),is_carpisma*0)
    not_carpisma = 1-is_carpisma
    print("carpisma timer",carpisma_timer_orig)
    print("is carpisma",is_carpisma.shape)
    print("not carpisma",not_carpisma.shape)
    not_carpisma = K.expand_dims(not_carpisma, axis=1)
    not_carpisma = K.repeat_elements(not_carpisma, action_count, axis=1)
    res_crossed = res_crossed * not_carpisma
    res = K.concatenate([res_crossed,carpisma_timer_orig],axis=1)
    return res

def check_carpisma(x):
    cars = tf.strided_slice(x, [0, 0, 0, 3], [1,210, 160, 40], [1, 1, 1, 4]) 
    cars = K.sum(cars,axis=3,keepdims=True)
    tavuks = tf.strided_slice(x, [0, 0, 0, 43], [1,210, 160, 52], [1, 1, 1, 4]) 
    tavuks = K.sum(tavuks,axis=3,keepdims=True)
    #cars = K.sum(x[:,:,:,3:4:40],axis=3,keepdims=True)
    #tavuks = K.sum(x[:,:,:,43:4:52],axis=3,keepdims=True)
    #cars_capped = K.switch(K.greater(cars, 0.1), K.ones_like(cars), K.zeros_like(cars))
    #tavuks_capped = K.switch(K.greater(tavuks, 0.1), K.ones_like(tavuks), K.zeros_like(tavuks))
    cars_capped = K.switch(cars>0.1,K.maximum(cars,1),cars*0)
    tavuks_capped = K.switch(tavuks>0.1,K.maximum(tavuks,1),tavuks*0)
    carpisma = K.sum(cars_capped * tavuks_capped, axis=[1,2,3])
    carpisma = K.minimum(carpisma, 1)
    return carpisma

def get_tavuk_alpha(x):
    tavuks = tf.strided_slice(x, [0, 0, 0, 43], [1,210, 160, 52], [1, 1, 1, 4]) 
    tavuks = K.sum(tavuks,axis=3,keepdims=True)
    #tavuks = K.sum(x[:,:,:,40:4:52],axis=3,keepdims=True)
    #tavuks = x[:,:,:,40:44]
    tavuks_capped = K.switch(tavuks>0.1,K.maximum(tavuks,1),tavuks*0)
    return tavuks_capped

def get_car_alpha(x):
    cars = tf.strided_slice(x, [0, 0, 0, 3], [1,210, 160, 40], [1, 1, 1, 4]) 
    cars = K.sum(cars,axis=3,keepdims=True)
    #cars = K.sum(x[:,:,:,0:4:40],axis=3,keepdims=True)
    #cars = x[:,:,:,0:4]
    cars_capped = K.switch(cars>0.1,K.maximum(cars,1),cars*0)
    return cars_capped
   
def create_timer(length,name):
    input = Input(shape=(length,1), name=name)
    conv = Conv1D(filters=1, kernel_size=3,strides=1,padding="same",use_bias=False,input_shape=(length,1))
    next = conv(input)
    w = conv.get_weights()
    w[0] = w[0] * 0
    w[0][0,0,0] = 1
    conv.set_weights(w)
    conv.trainable = False
    is_active = Lambda(is_crossed_fn)(input)    
    return input, next, is_active
    
def reward_sum(x):
    res = K.squeeze(x,axis=3)
    res = K.squeeze(res,axis=2)
    #res = K.squeeze(res,axis=1)
    return res

def get_transcoder(input_shape, action_count, conv_info):
    cross_timer_input, cross_timer_next, is_crossed = create_timer(cross_timer_count, name='cross_timer_input')
    carpisma_timer_input, carpisma_timer_next, is_carpisma = create_timer(carpisma_timer_count, name='carpisma_timer_input')

    input = Input(shape=input_shape, name='encoded_input')
    input_action = Input(shape=(action_count,), name='action')
    input_action_crossed = Lambda(merge_action_crossed_fn)([input_action,is_crossed,carpisma_timer_input])

    merged_input = Lambda(action_merge)([input, input_action_crossed]) #input_action
    next_state = merged_input
    idx = 0
    for info in conv_info:
        if info[0] == Conv2D:
            conv = info[0](filters=info[1],kernel_size=info[2],strides=info[3],padding="same",activation=info[4],use_bias=False,kernel_initializer=tf.keras.initializers.RandomUniform(minval=0.005, maxval=0.01, seed=None),name=f'conv_transcoder_{idx}') #, kernel_initializer=RandomNormal(mean=0.0, stddev=0.001, seed=None), kernel_regularizer=l1(0.1), bias_regularizer=l1(0.1)
        elif info[0] == Reshape:
            conv = info[0]((info[1],info[2],info[3]))
        elif info[0] == Conv2DTranspose:
            conv = info[0](filters=info[1],kernel_size=info[2],strides=info[3],padding="valid",activation=info[4],use_bias=False,kernel_initializer='glorot_normal',name=f'conv_transcoder_{idx}')
        idx += 1
        next_state = conv(next_state)
    tavuk = Lambda(channel_slice_function(10,13))(next_state)
    cars = Lambda(channel_slice_function(0,10))(next_state)
    x0 = 46
    y0 = 190
    top_crop = 18 #20 @ersin - en ustte kaliyordu
    bottom_crop = 17

    tavuk_top = Lambda(image_slice_function(((0,top_crop),(0,160))))(tavuk)
    tavuk_top_sum = Lambda(sum_tavuk)(tavuk_top)
    reward = Lambda(reward_sum)(tavuk_top_sum)
    cross_timer_next = Lambda(add_to_timer(cross_timer_count))([cross_timer_next, tavuk_top_sum])
    tavuk_bottom = Lambda(image_slice_function(((210-bottom_crop,210),(0,160))))(tavuk)
    tavuk_bottom_sum = Lambda(sum_tavuk)(tavuk_bottom)
    tavuk_middle = Lambda(image_slice_function(((top_crop,210-bottom_crop),(0,160))))(tavuk)
    tavuk_add_bottom = ZeroPadding2D(((y0,210-y0-1),(x0,160-x0-1)))(tavuk_top_sum)
    tavuk_add_bottom2 = ZeroPadding2D(((y0,210-y0-1),(x0,160-x0-1)))(tavuk_bottom_sum)
    tavuk_middle_padded = ZeroPadding2D(((top_crop,bottom_crop),(0,0)))(tavuk_middle)
    tavuk2 = Lambda(channel_slice_function(0,1))(tavuk_middle_padded)
    tavuk2 = Add()([tavuk_add_bottom, tavuk_add_bottom2, tavuk2])
    tavuk_rest = Lambda(channel_slice_function(1,3))(tavuk_middle_padded)
    next_state = Concatenate()([cars, tavuk2, tavuk_rest])
    next_obs = model_ae_decoder_channels(next_state)
    carpisma = Lambda(check_carpisma)(next_obs)
    #tavuk_alpha = Lambda(get_tavuk_alpha)(next_obs)
    #car_alpha = Lambda(get_car_alpha)(next_obs)
    carpisma_timer_next = Lambda(add_to_timer(carpisma_timer_count))([carpisma_timer_next, carpisma])
    model_transcoder = Model(inputs = [input, input_action, cross_timer_input, carpisma_timer_input], outputs = [next_state, cross_timer_next, carpisma_timer_next,reward], name="transcoder") #,car_alpha,tavuk_alpha
    return model_transcoder

#(64,4),(64,4),
conv_trans_info = [(Conv2D,cc,(9,9),1,'relu')]
model_transcoder = get_transcoder(input_shape_orig, env.action_space.n, conv_trans_info)

observations = Input(shape=(input_shape_orig[0],input_shape_orig[1], 4), name="observation3_large")
input_action = Input(shape=(env.action_space.n,), name='action')
input_cross_timer = Input(shape=(cross_timer_count,1), name='cross_timer_input3')
input_carpisma_timer = Input(shape=(carpisma_timer_count,1), name='carpisma_timer_input3')

state = model_ae_encoder(observations)
next_state = model_transcoder([state,input_action,input_cross_timer,input_carpisma_timer])
#predicted = model_ae_decoder_channels(next_state)
predicted = model_ae_decoder(next_state[0])
model = Model(inputs=[observations,input_action,input_cross_timer,input_carpisma_timer], outputs=[predicted, next_state[1]])

#optimizer = Adagrad(lr=0.01, epsilon=None, decay=0.0)
#optimizer = Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.0)
#optimizer = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
#optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
#optimizer = Nadam(lr=0.0001)
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# @derya, epsilon ve decay'ler icin farkli default degerlerin oldugu site adresi iste
# amsgrad dene

def custom_loss(y_true, y_pred):
    return K.mean(K.square(y_pred[:,10:-10,10:-10,:] - y_true[:,10:-10,10:-10,:])) # her taraftan 10 kirp
    #return K.mean(K.square(y_pred[0][:,10:-10,10:-10,:] - y_true[0][:,10:-10,10:-10,:])) # her taraftan 10 kirp
    #return K.mean(K.square(y_pred[:,10:-10,10:-10,0:10] - y_true[:,10:-10,10:-10,0:10])) # her taraftan 10 kirp
    #return K.mean(K.square(y_pred - y_true))
    #return K.mean(K.square(y_pred[:,:,:,3:7] - y_true[:,:,:,3:7]))
    #return K.mean(K.switch(K.greater(y_true, 0.8), K.square(y_pred - y_true), K.abs(y_pred - y_true)))
    #return K.mean(K.switch(K.greater(y_true, 0.8), 100*K.square(y_pred - y_true), K.square(y_pred - y_true)))
    #return K.mean(y_pred)


w = model_transcoder.get_weights()
print('weight length', len(w), w[0].shape)
w[0] = w[0] * 0
##15,15,52,13
#w[0][16,19,0,0] = 0.66667
#w[0][16,20,0,0] = 0.33333
#w[0][16,20,1,1] = 1
#w[0][16,21,2,2] = 1
#w[0][16,24,3,3] = 1
#w[0][16,32,4,4] = 1
#w[0][16,0,5,5] = 1
#w[0][16,8,6,6] = 1
#w[0][16,11,7,7] = 1
#w[0][16,12,8,8] = 1
#w[0][16,12,9,9] = 0.33333
#w[0][16,13,9,9] = 0.66667
#w[0][16,16,13+10,10] = 1
#w[0][16,16,13+12,12] = 1
#w[0][32,16,13*2+10,10] = 1
#w[0][32,16,13*2+12,12] = 1
#w[0][0,16,13*3+10,10] = 1
#w[0][0,16,13*3+12,12] = 1
#w[0][16,16,11,11] = 1

#0111110 - 0.83333 0.16666
#1111111   1
#1211121   0.75 0.25
#2222222   1
#4444444   1
#4444444 
#2222222
#1211121
#1111111
#0111110
#4

w[0][4,5,0,0] = 0.8000
w[0][4,4,0,0] = 0.2000
w[0][4,5,1,1] = 1
w[0][4,5,2,2] = 0.66666
w[0][4,6,2,2] = 0.33334
w[0][4,6,3,3] = 1
w[0][4,8,4,4] = 1
w[0][4,0,5,5] = 1
w[0][4,2,6,6] = 1
w[0][4,3,7,7] = 0.66666
w[0][4,2,7,7] = 0.33334
w[0][4,3,8,8] = 1
w[0][4,3,9,9] = 0.8000
w[0][4,4,9,9] = 0.2000
w[0][4,4,13+10,10] = 1
w[0][4,4,13+12,12] = 1
w[0][8,4,13*2+12,10] = 1
w[0][8,4,13*2+10,12] = 1
w[0][0,4,13*3+12,10] = 1
w[0][0,4,13*3+10,12] = 1
w[0][4,4,13+11,10] = 1
w[0][6,4,13*2+11,10] = 1
w[0][2,4,13*3+11,10] = 1

action_count = env.action_space.n
for I in range(carpisma_timer_count):
    if I<6:
        w[0][0,4,13*(I+action_count+1)+10,11] = 1
        w[0][0,4,13*(I+action_count+1)+12,11] = 1
        w[0][0,4,13*(I+action_count+1)+11,10] = 1
    else:
        w[0][4,4,13*(I+action_count+1)+10,10] = 1
        w[0][4,4,13*(I+action_count+1)+12,12] = 1
        w[0][4,4,13*(I+action_count+1)+11,11] = 1

#
model_transcoder.set_weights(w)
for layer in model_transcoder.layers:
	layer.trainable = False
    
model_transcoder.compile(optimizer=optimizer,loss=[custom_loss,'mse','mse','mse'],loss_weights=[1,0,0,0]) #custom_loss, 'mse'
for layer in model.layers:
    print(layer.name)
    if not layer.name.startswith('transcoder'):
        print('not trainable')
        layer.trainable = False
    else:
        print('trainable')

def save_image(outmap, batch_size, fname):
    output_image = np.copy(outmap)
    output_image[output_image>1] = 1
    output_image[output_image<0] = 0
    output_image = output_image * 255.0
    #outmap[outmap>1] = 1
    #outmap[outmap<0] = 0
    #if len(outmap.shape) == 4:
    #	output_image = outmap[:,:,:,:] * 255.0
    #else:
    #	output_image = outmap[:,:,:] * 255.0
    output_image = output_image.astype(np.uint8)
    if output_image.shape[3] == 1:
        output_image = np.squeeze(output_image,axis=3)
    for J in range(batch_size):
        if len(output_image.shape) == 4:
            output_im = Image.fromarray(output_image[J,:,:,:])
        else:
            output_im = Image.fromarray(output_image[J,:,:])
        output_im.save(fname + f'_{J}.png')

bg = np.array(Image.open(f'images/bg.png'))

    
ACTION_REPEAT = 1
def gameStep(action):
    tot_reward = 0
    for I in range(ACTION_REPEAT):
        ob, reward, done, _ = env.step(action)
        tot_reward += reward
        if done:
            break
    return ob, tot_reward, done

factions = None
if not output_path is None:
    factions = open(f"{output_path}actions.txt","w") 

batch_size = 1
NO_OP = 10
#env = gym.wrappers.Monitor(env, '/raid/users/ebasaran/src/model-based-rl/src/tests/cartpole-experiment-1', force=True)
last_ob = env.reset()
for I in range(NO_OP):
    last_ob = gameStep(env.action_space.sample())
last_ob = last_ob[0]
last_ob = remove_bg(last_ob, bg)
batch_prev = []
batch_next = []
actions = []
batch_id = 0
cross_timer = np.zeros((batch_size,cross_timer_count,1))
carpisma_timer = np.zeros((batch_size,carpisma_timer_count,1))
reward = np.zeros((batch_size,))
#car_alpha = np.zeros((batch_size,210,160,1))
#tavuk_alpha = np.zeros((batch_size,210,160,1))
for I in range(1000):
    #ob = env.step(env.action_space.sample()) # take a random action
    #action = env.action_space.sample()
    #action = 1
    action = np.random.choice(range(0,env.action_space.n), p=[0.2,0.6,0.2])
    ob = gameStep(action)
    #Image.fromarray(ob[0]).save(f'{output_path}step_{I}_{action}.png')
    done = ob[2]
    if done:
        last_ob = env.reset()
        last_ob = remove_bg(last_ob, bg)
    else:
        ob = ob[0]
        #Image.fromarray((ob).astype(np.uint8)).save(f'{output_path}step_{I}.png')
        ob = remove_bg(ob, bg)
        batch_prev.append(last_ob)
        batch_next.append(ob)
        actions.append(action)
        if len(batch_prev) == batch_size:
            state = model_ae_encoder.predict_on_batch([batch_prev])
            next_state = model_ae_encoder.predict_on_batch([batch_next])
            ####next_channels = model_ae_decoder_channels.predict_on_batch(next_state)
            ####res = model.train_on_batch([batch_prev], [next_channels])
            ####res = model_transcoder.train_on_batch([state], [state])
            res = None
            res = model_transcoder.train_on_batch([state, to_categorical(np.array(actions), num_classes=env.action_space.n), cross_timer, carpisma_timer], [next_state, cross_timer, carpisma_timer,reward]) #car_alpha, tavuk_alpha
            res = res[0]
            if not sw is None and not res is None:
                sw.add([res], batch_id)
            print(batch_id, res,np.squeeze(carpisma_timer),np.squeeze(reward))
            #if batch_id % 100 == 0:
            #	ws = model_transcoder.get_weights()
            #	for I in range(len(ws)):
            #		w = ws[I]
            #		w[np.abs(w)<0.01] = 0
            #	model_transcoder.set_weights(ws)
            if batch_id%1 == 0 and not output_path is None:
                current_action = to_categorical(np.array(actions), num_classes=env.action_space.n)
                
                next_est = model_transcoder.predict_on_batch([state, current_action, cross_timer, carpisma_timer])
                cross_timer = next_est[1]
                carpisma_timer = next_est[2]
                reward = next_est[3]
                #for K in range(cc):
                #	save_image(state[:,:,:,K], 1, f'{output_path}{batch_id}_{0}_encoded_current_{K}')
                #	save_image(next_est[:,:,:,K], 1, f'{output_path}{batch_id}_{0}_encoded_est_{K}')
                #	save_image(next_state[:,:,:,K], 1, f'{output_path}{batch_id}_{0}_encoded_next_{K}')
                #save_image(next_est[3], batch_size, f'{output_path}{batch_id}_car_alpha')
                #save_image(next_est[4], batch_size, f'{output_path}{batch_id}_tavuk_alpha')
                if batch_id%100 == 0:
                    model_transcoder.save(f'{output_path}{I}_model_transcoder.h5')
                
                next_obs = model_ae_decoder.predict_on_batch([next_est[0]])
                next_obs = next_obs * 255.0
                next_obs[next_obs>255] = 255
                next_obs[next_obs<0] = 0
                next_obs = next_obs.astype(np.uint8)
                for J in range(batch_size):
                    if not factions is None:
                        factions.write(f"{batch_id}\t{J}\t{actions[J]}\n")
                    Image.fromarray((batch_next[J]*255).astype(np.uint8)).save(f'{output_path}{batch_id}_{J}_real.png')
                    Image.fromarray(next_obs[J,:,:,:]).save(f'{output_path}{batch_id}_{J}_estimated.png')
            #if batch_id%100 == 0:
            #	model_transcoder.save(f'{output_path}{I}_model_transcoder.h5')

            batch_id+=1
            batch_prev = []
            batch_next = []
            actions = []
        last_ob = ob

env.close()


