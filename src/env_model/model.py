from runner.runner import *
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Add, Subtract, Lambda, Multiply, Flatten, Dropout, GaussianNoise
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Nadam
from tensorflow.keras.initializers import RandomUniform, Constant
from nets.layers import printLayer #RBFLayer, 
import numpy as np 
import tensorflow.keras
import tensorflow.keras.backend as K
from nets.loss import huber_loss, huber_loss_mse
from nets.optimizers import DqnRMSprop

from utils.summary_writer import SummaryWriter

class EnvModel(object):
    def __init__(self, ops):
        self.ops = ops
        pass
    def predict_next(self, current_state, action):
        pass
    def train_next(self, current_state, action):
        pass

class EnvOps(object):
    def __init__(self, input_size, action_count, learning_rate, mode = "test"):
        self.INPUT_SIZE = input_size
        self.ACTION_COUNT = action_count
        self.LEARNING_RATE = learning_rate
        self.mode = mode
        
class EnvModelCartpole(EnvModel):
    def __init__(self, ops):
        super(EnvModelCartpole, self).__init__(ops)
        self.model = self.get_model()
    def get_model(self):
        input_shape=self.ops.INPUT_SIZE
        input = Input(shape=input_shape, name='observation')
        x = input
        x = Dense(256,activation="relu", kernel_initializer='he_uniform')(x)
        x = Dense(256,activation="relu", kernel_initializer='he_uniform')(x)
        action_outputs = []
        done_outputs = []
        #reward_outputs = []
        reward_output = Dense(self.ops.ACTION_COUNT, kernel_initializer='he_uniform')(x)
        losses = []
        loss_weights = []
        for I in range(self.ops.ACTION_COUNT):
            action_output = Dense(input_shape[0], kernel_initializer='he_uniform')(x)
            action_outputs.append(action_output)
            done_output = Dense(1, kernel_initializer='he_uniform', activation='sigmoid')(x)
            done_outputs.append(done_output)
            losses.append('mse')
            loss_weights.append(1)
            #reward_hidden = Dense(256,activation="relu", kernel_initializer='he_uniform')(x)
            #reward_output = Dense(1, kernel_initializer=keras.initializers.random_uniform(-0.02, 0.02))(reward_hidden)
            #reward_outputs.append(reward_output)
        losses.append('mse')
        loss_weights.append(1)
        for I in range(self.ops.ACTION_COUNT):
            losses.append('binary_crossentropy')
            loss_weights.append(0.3)
        model = Model(inputs=[input], outputs=action_outputs + [reward_output] + done_outputs)
        my_optimizer = Adam(lr=self.ops.LEARNING_RATE)
        #my_optimizer = RMSprop(lr=self.ops.LEARNING_RATE, rho=0.90, decay=0.0) #epsilon=None, 
        model.compile(optimizer=my_optimizer,loss=losses, loss_weights=loss_weights)
        return model
    def predict_next(self, current_state):
        return self.model.predict(np.array(current_state, dtype='f'))
    def train_next(self, current_state, next_states):
        return self.model.train_on_batch(np.array(current_state, dtype='f'), next_states)

class EnvModelLine(EnvModel):
    def __init__(self, ops):
        super(EnvModelLine, self).__init__(ops)
        self.model = self.get_model()
    def get_model(self):
        input_shape=self.ops.INPUT_SIZE
        input = Input(shape=input_shape, name='observation')
        x = input
        x = Dense(24,activation="relu", kernel_initializer='he_uniform')(x)
        action_outputs = []
        done_outputs = []
        reward_outputs = []
        losses = []
        loss_weights = []
        for I in range(self.ops.ACTION_COUNT):
            action_output = Dense(input_shape[0], kernel_initializer='he_uniform')(x)
            action_outputs.append(action_output)
            done_output = Dense(1, kernel_initializer='he_uniform', activation='sigmoid')(x)
            done_outputs.append(done_output)
            losses.append('mse')
            loss_weights.append(1)
            #reward_output = Dense(20, activation='sigmoid')(input)
            #reward_output = RBFLayer(100, initializer=RandomUniform(-1.0, 1.0), betas=0.02)(input)
            reward_output = Dense(20, activation='relu')(input)
            #reward_output = Dense(20, activation='hard_sigmoid')(input)
            reward_output = Dense(1)(reward_output)
            reward_outputs.append(reward_output)
            #reward_hidden = Dense(256,activation="relu", kernel_initializer='he_uniform')(x)
            #reward_output = Dense(1, kernel_initializer=keras.initializers.random_uniform(-0.02, 0.02))(reward_hidden)
            #reward_outputs.append(reward_output)
        losses.append('mse')
        loss_weights.append(5)
        reward_output_concat = Concatenate()(reward_outputs)
        for I in range(self.ops.ACTION_COUNT):
            losses.append('binary_crossentropy')
            loss_weights.append(0.3)
        model = Model(inputs=[input], outputs=action_outputs + [reward_output_concat] + done_outputs)
        my_optimizer = Adam(lr=self.ops.LEARNING_RATE)
        #my_optimizer = SGD(lr=self.ops.LEARNING_RATE)
        #my_optimizer = RMSprop(lr=self.ops.LEARNING_RATE) #epsilon=None, , rho=0.90, decay=0.0
        model.compile(optimizer=my_optimizer,loss=losses, loss_weights=loss_weights)
        return model
    def predict_next(self, current_state):
        return self.model.predict(np.array(current_state, dtype='f'))
    def train_next(self, current_state, next_states):
        return self.model.train_on_batch(np.array(current_state, dtype='f'), next_states)	
        
class EnvModelLineManual(EnvModel):
    def __init__(self, ops):
        super(EnvModelLineManual, self).__init__(ops)
        self.model = self.get_model()
    def get_model(self):
        input_shape=self.ops.INPUT_SIZE
        input = Input(shape=input_shape, name='observation')
        x = input
        #x = Dense(24,activation="relu", kernel_initializer='he_uniform')(x)
        action_outputs = []
        done_outputs = []
        reward_outputs = []
        losses = []
        loss_weights = []
        for I in range(self.ops.ACTION_COUNT):
            action_output = Dense(input_shape[0], kernel_initializer=Constant(1), bias_initializer=Constant(0.05 if I==0 else -0.05))(x)
            action_output = Lambda(lambda x: K.clip(x, -1, 1))(action_output)
            action_outputs.append(action_output)
            print(action_output.shape)
            done_output = Lambda(lambda x: K.switch(K.abs(x - 0.5) < 0.05, 1+x*0, x*0))(action_output)
            #done_output = Dense(1, kernel_initializer='he_uniform', activation='sigmoid')(x)
            done_outputs.append(done_output)
            losses.append('mse')
            loss_weights.append(1)
            #reward_output = Dense(20, activation='sigmoid')(input)
            #reward_output = RBFLayer(100, initializer=RandomUniform(-1.0, 1.0), betas=0.02)(input)
            #reward_output = Dense(20, activation='relu')(input)
            #reward_output = Dense(20, activation='hard_sigmoid')(input)
            #reward_output = Dense(1)(reward_output)
            reward_output = Lambda(lambda x: K.switch(x < 0.01, x*0 -0.01, x*0 +1 ))(done_output)
            reward_outputs.append(reward_output)
            #reward_hidden = Dense(256,activation="relu", kernel_initializer='he_uniform')(x)
            #reward_output = Dense(1, kernel_initializer=keras.initializers.random_uniform(-0.02, 0.02))(reward_hidden)
            #reward_outputs.append(reward_output)
        losses.append('mse')
        loss_weights.append(5)
        reward_output_concat = Concatenate()(reward_outputs)
        for I in range(self.ops.ACTION_COUNT):
            losses.append('binary_crossentropy')
            loss_weights.append(0.3)
        model = Model(inputs=[input], outputs=action_outputs + [reward_output_concat] + done_outputs)
        my_optimizer = Adam(lr=self.ops.LEARNING_RATE)
        #my_optimizer = SGD(lr=self.ops.LEARNING_RATE)
        #my_optimizer = RMSprop(lr=self.ops.LEARNING_RATE) #epsilon=None, , rho=0.90, decay=0.0
        model.compile(optimizer=my_optimizer,loss=losses, loss_weights=loss_weights)
        return model
    def predict_next(self, current_state):
        return self.model.predict(np.array(current_state, dtype='f'))
    def train_next(self, current_state, next_states):
        return self.model.train_on_batch(np.array(current_state, dtype='f'), next_states)	

# adapted from https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

def cartpole_next_state_fn(input, action):
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = (masspole + masscart)
    length = 0.5 # actually half the pole's length
    polemass_length = (masspole * length)
    force_mag = 10.0
    tau = 0.02  # seconds between state updates
    x = input[:,0]
    x_dot = input[:,1]
    theta = input[:,2]
    theta_dot = input[:,3]
    costheta = K.cos(theta)
    sintheta = K.sin(theta)
    force = force_mag if action == 1 else -force_mag
    temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta* temp) / (length * (4.0/3.0 - masspole * costheta * costheta / total_mass))
    xacc  = temp - polemass_length * thetaacc * costheta / total_mass
    x  = x + tau * x_dot
    x_dot = x_dot + tau * xacc
    theta = theta + tau * theta_dot
    theta_dot = theta_dot + tau * thetaacc
    return K.stack([x, x_dot, theta, theta_dot], axis=1)

import math 

class EnvModelCartPoleManual(EnvModel):
    def __init__(self, ops):
        super(EnvModelCartPoleManual, self).__init__(ops)
        self.model = self.get_model()
        
    def get_model(self):
        input_shape=self.ops.INPUT_SIZE
        input = Input(shape=input_shape, name='observation')

        action_outputs = []
        done_outputs = []
        reward_outputs = []

        theta_threshold_radians = 12 * 2 * math.pi / 360
        x_threshold = 2.4
        
        for I in range(self.ops.ACTION_COUNT):
            action_output = Lambda(cartpole_next_state_fn, arguments={'action': I})(input)
            action_outputs.append(action_output)
            print(action_output.shape)
            x_threshold_test = Lambda(lambda x: K.switch(K.abs(x[:,0]) > x_threshold, 1+x[:,0:1]*0, x[:,0:1]*0))(action_output)
            theta_threshold_test = Lambda(lambda x: K.switch(K.abs(x[:,2]) > theta_threshold_radians, 1+x[:,0:1]*0, x[:,0:1]*0))(action_output)
            print(x_threshold_test.shape)
            print(theta_threshold_test.shape)
            done_output = Lambda(lambda x: K.switch(x[0]+x[1], 1+x[0]*0, x[0]*0))([x_threshold_test, theta_threshold_test])
            done_outputs.append(done_output)
            # @ersin: -100: penalizer'li hali
            reward_output = Lambda(lambda x: K.switch(x < 0.01, x[:,0:1]*0 + 1, x[:,0:1]*0 -5 ))(done_output)
            reward_outputs.append(reward_output)
        reward_output_concat = Concatenate()(reward_outputs)
        model = Model(inputs=[input], outputs=action_outputs + [reward_output_concat] + done_outputs)
        my_optimizer = Adam(lr=self.ops.LEARNING_RATE)
        model.compile(optimizer=my_optimizer,loss='mse')
        return model
    def predict_next(self, current_state):
        return self.model.predict(np.array(current_state, dtype='f'))
    def train_next(self, current_state, next_states):
        return self.model.train_on_batch(np.array(current_state, dtype='f'), next_states)	
                
# adapted from https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py

def mountaincar_next_state_fn(input, action):
    min_position = -1.2
    max_position = 0.6
    max_speed = 0.07
    goal_position = 0.5

    position = input[:,0]
    velocity = input[:,1]

    velocity += (action-1)*0.001 + K.cos(3*position)*(-0.0025)
    velocity = K.clip(velocity, -max_speed, max_speed)
    position += velocity
    position = K.clip(position, min_position, max_position)
    #if (position==min_position and velocity<0): velocity = 0
    #print(position, position.shape)
    #print(velocity, velocity.shape)
    velocity = K.switch(K.equal(position, min_position), K.switch(velocity<0, velocity*0,velocity), velocity) #
    return K.stack([position, velocity], axis=1)

class EnvModelMountainCarManual(EnvModel):
    def __init__(self, ops):
        super(EnvModelMountainCarManual, self).__init__(ops)
        self.model = self.get_model()
        
    def get_model(self):
        input_shape=self.ops.INPUT_SIZE
        input = Input(shape=input_shape, name='observation')

        action_outputs = []
        done_outputs = []
        reward_outputs = []

        goal_position = 0.5
        
        for I in range(self.ops.ACTION_COUNT):
            action_output = Lambda(mountaincar_next_state_fn, arguments={'action': I})(input)
            action_outputs.append(action_output)
            #print(action_output.shape)
            done_output = Lambda(lambda x: K.switch(x[:,0] > goal_position, 1+x[:,0:1]*0, x[:,0:1]*0))(action_output)
            done_outputs.append(done_output)
            # @ersin: -100: penalizer'li hali
            #reward_output = Lambda(lambda x: x[:,0:1]*0 - 1)(done_output)
            reward_output = Lambda(lambda x: K.switch(x < 0.01, x[:,0:1]*0 -0.01, x[:,0:1]*0 + 1 ))(done_output)

            reward_outputs.append(reward_output)
        reward_output_concat = Concatenate()(reward_outputs)
        model = Model(inputs=[input], outputs=action_outputs + [reward_output_concat] + done_outputs)
        my_optimizer = Adam(lr=self.ops.LEARNING_RATE)
        model.compile(optimizer=my_optimizer,loss='mse')
        return model
    def predict_next(self, current_state):
        return self.model.predict(np.array(current_state, dtype='f'))
    def train_next(self, current_state, next_states):
        return self.model.train_on_batch(np.array(current_state, dtype='f'), next_states)			
        
class EnvModelLunarLander(EnvModel):
    def __init__(self, ops):
        super(EnvModelLunarLander, self).__init__(ops)
        self.model = self.get_model()
    def get_model(self):
        input_shape=self.ops.INPUT_SIZE
        input = Input(shape=input_shape, name='observation')
        x = input
        x = Dense(256,activation="relu")(x)
        x = Dense(256,activation="relu")(x)
        action_outputs = []
        done_outputs = []
        #reward_outputs = []
        c = Dense(256,activation="relu")(x)
        reward_output = Dense(self.ops.ACTION_COUNT)(c)
        losses = []
        loss_weights = []
        for I in range(self.ops.ACTION_COUNT):
            a = Dense(256,activation="relu")(x)
            action_output = Dense(input_shape[0])(a)
            action_outputs.append(action_output)
            b = Dense(256,activation="relu")(x)
            done_output = Dense(1, activation='sigmoid')(b)
            done_outputs.append(done_output)
            losses.append('mse')
            loss_weights.append(1) # next state loss weight
            #reward_hidden = Dense(256,activation="relu", kernel_initializer='he_uniform')(x)
            #reward_output = Dense(1, kernel_initializer=keras.initializers.random_uniform(-0.02, 0.02))(reward_hidden)
            #reward_outputs.append(reward_output)
        losses.append('mse')
        loss_weights.append(0.3) #reward loss weight
        for I in range(self.ops.ACTION_COUNT):
            losses.append('binary_crossentropy')
            loss_weights.append(0.3) #done loss weight
        model = Model(inputs=[input], outputs=action_outputs + [reward_output] + done_outputs)
        my_optimizer = Adam(lr=self.ops.LEARNING_RATE)
        #my_optimizer = RMSprop(lr=self.ops.LEARNING_RATE, rho=0.90, decay=0.0) #epsilon=None, 
        model.compile(optimizer=my_optimizer,loss=losses, loss_weights=loss_weights)
        return model
    def predict_next(self, current_state):
        return self.model.predict(np.array(current_state, dtype='f'))
    def train_next(self, current_state, next_states):
        return self.model.train_on_batch(np.array(current_state, dtype='f'), next_states)
        
class EnvModelLunarLander2(EnvModelLunarLander):
    def get_model(self):
        input_shape=self.ops.INPUT_SIZE
        input = Input(shape=input_shape, name='observation')
        rx = input
        ax = input
        dx = input
        action_outputs = []
        done_outputs = []
        #reward_outputs = []
        rx = Dense(1024,activation="sigmoid")(rx)
        rx = Dense(1024,activation="sigmoid")(rx)
        reward_output = Dense(self.ops.ACTION_COUNT)(rx)
        losses = []
        loss_weights = []
        for I in range(self.ops.ACTION_COUNT):
            ax = Dense(1024,activation="sigmoid")(ax)
            ax = Dense(1024,activation="sigmoid")(ax)
            action_output = Dense(input_shape[0])(ax)
            action_outputs.append(action_output)
            dx = Dense(1024,activation="sigmoid")(dx)
            dx = Dense(1024,activation="sigmoid")(dx)
            done_output = Dense(1, activation='sigmoid')(dx)
            done_outputs.append(done_output)
            losses.append('mse')
            loss_weights.append(1) # next state loss weight
            #reward_hidden = Dense(256,activation="relu", kernel_initializer='he_uniform')(x)
            #reward_output = Dense(1, kernel_initializer=keras.initializers.random_uniform(-0.02, 0.02))(reward_hidden)
            #reward_outputs.append(reward_output)
        losses.append('mse')
        loss_weights.append(0.3) #reward loss weight
        for I in range(self.ops.ACTION_COUNT):
            losses.append('binary_crossentropy')
            loss_weights.append(0.3) #done loss weight
        model = Model(inputs=[input], outputs=action_outputs + [reward_output] + done_outputs)
        my_optimizer = Adam(lr=self.ops.LEARNING_RATE)
        #my_optimizer = RMSprop(lr=self.ops.LEARNING_RATE, rho=0.90, decay=0.0) #epsilon=None, 
        model.compile(optimizer=my_optimizer,loss=losses, loss_weights=loss_weights)
        return model
        
def custom_loss(y_true, y_pred):
    return K.mean(K.square(y_pred[:,10:-10,10:-10,:] - y_true[:,10:-10,10:-10,:])) # her taraftan 10 kirp
        
class EnvModelFreewayManual(EnvModel):
    def __init__(self, ops):
        super(EnvModelFreewayManual, self).__init__(ops)
        self.model = self.get_model()
    def get_model(self):
        # input, input_action, cross_timer_input, carpisma_timer_input
        #self.model_encoder = tf.keras.models.load_model(f'tests/freeway/test_run_68/6000_model_large_encoder.h5')
        self.model_transcoder = tf.keras.models.load_model(f'tests/freeway/test_run_m150/0_model_transcoder.h5', custom_objects={'tf': tf, 'custom_loss': custom_loss})
        inputs = self.model_transcoder.inputs
        # next_state, cross_timer_next, carpisma_timer_next,reward
        outputs = self.model_transcoder.outputs
        
        cars_pos = [31, 47, 63, 79, 95, 111, 127, 143, 159, 175]
        tavuk_pos = 46
        
        def to_transcoder_input(x):
            layers = []
            for I in range(10):
                tmp = K.expand_dims(x[1][:,:,I:I+1],axis=1)
                tmp = K.spatial_2d_padding(tmp,((cars_pos[I],210-cars_pos[I]-1),(0,0)))
                layers.append(tmp)
            for I in range(3):
                tmp = K.expand_dims(x[0][:,:,I:I+1],axis=2)
                tmp = K.spatial_2d_padding(tmp,((0,0),(tavuk_pos,160-tavuk_pos-1)))
                layers.append(tmp)
            print([l.shape for l in layers])
            return K.concatenate(layers,axis=3)
        def from_transcoder_output_tavuk(x):
            tmp = x[:,:,tavuk_pos:tavuk_pos+1,10:13]
            tmp = K.squeeze(tmp,axis=2)
            return tmp
        def from_transcoder_output_car(x):
            layers = []
            for I in range(10):
                tmp = x[:,cars_pos[I]:cars_pos[I]+1,:,I:I+1]
                tmp = K.squeeze(tmp, axis=1)
                layers.append(tmp)
            return K.concatenate(layers, axis=2)
        
        input_tavuk = Input(shape=(210,3),name='input_tavuk')
        input_car = Input(shape=(160,10),name='input_car')
        input = Lambda(to_transcoder_input)([input_tavuk, input_car])
        cross_timer_input = Input(shape=inputs[2].shape[1:], name='cross_timer_input')
        carpisma_timer_input = Input(shape=inputs[3].shape[1:], name='carpisma_timer_input')
        
        action_outputs = []
        done_outputs = []
        reward_outputs = []

        action_count = inputs[1].shape[1]
        def tile_action_batch(I):
            def tile_input(x):
                ain = np.zeros((1,action_count),dtype="f")
                ain[0,I] = 1
                co = K.constant(ain)
                return K.tile(co, (K.shape(x)[0], 1))
            return tile_input
                
        print("ACTION_COUNT", action_count)
        for I in range(action_count):
            input_action = Lambda(tile_action_batch(I),name=f'input_action_tiled_{I}')(input)
            print("Transcoder Input shapes", input.shape, input_action.shape, cross_timer_input.shape, carpisma_timer_input.shape)
            layer_outputs = self.model_transcoder([input, input_action, cross_timer_input, carpisma_timer_input])
            tavuk_output = Lambda(from_transcoder_output_tavuk)(layer_outputs[0])
            car_output = Lambda(from_transcoder_output_car)(layer_outputs[0])
            action_outputs.extend([car_output, tavuk_output] + layer_outputs[1:3])
            #print(action_output.shape)
            reward_output = layer_outputs[3]
            reward_outputs.append(reward_output)
            done_output = Lambda(lambda x: x*0)(reward_output)
            done_outputs.append(done_output)
        reward_output_concat = Concatenate()(reward_outputs)
        print(action_outputs+ [reward_output_concat]+ done_outputs)
        model = Model(inputs=[input_car, input_tavuk,cross_timer_input,carpisma_timer_input], outputs=action_outputs + [reward_output_concat] + done_outputs, name='freeway_model')
        my_optimizer = Adam(lr=0.001)
        model.compile(optimizer=my_optimizer,loss='mse')
        return model
    def get_samples(self, N):
        def sampleCar():
            arr = np.zeros((N,160,10))
            col1 = np.repeat(np.arange(N),10)
            col2 = np.random.randint(0,160,N*10)
            col3 = np.transpose(np.repeat(np.resize(np.arange(10), (10,1)), N, axis=1)).flatten()
            arr[col1,col2,col3] = 1
            return arr
        def sampleTavuk():
            arr = np.zeros((N,210,3))
            col1 = np.arange(N)
            col2 = np.random.randint(22,190+1,N)
            col3 = np.random.randint(0,3,N)
            arr[col1,col2,col3] = 1
            return arr
        def sampleTimer(size,p=0.1):
            arr = np.zeros((N,size,1))
            col1 = np.random.random((N))
            col1 = col1<p
            count = np.count_nonzero(col1)
            col2 = np.random.randint(0,size,(count))
            arr[col1,col2] = 1
            return arr
        return [sampleCar(),sampleTavuk(),sampleTimer(7,0.2),sampleTimer(13,0.5)]
    def predict_next(self, current_state):
        return self.model.predict(current_state)
    def train_next(self, current_state, next_states):
        return self.model.train_on_batch(current_state, next_states)	
        
class EnvLearner(RunnerListener):
    def __init__(self, sampler, model, sw = None, reward_scale = 1):
        super(RunnerListener, self).__init__()
        self.sampler = sampler
        self.model = model
        label_action_losses = ['loss_action_' + str(a+1) for a in range(self.model.ops.ACTION_COUNT)]
        label_action_dones = ['loss_done_' + str(a+1) for a in range(self.model.ops.ACTION_COUNT)]
        self.sw = SummaryWriter(sw, ['Total loss'] + label_action_losses + ['loss_r'] + label_action_dones)
        self.total_step_count = 0
        self.reward_scale = reward_scale
    def on_step(self, ob, action, next_ob, reward, done):
        self.total_step_count += 1
        if self.sampler.has_sample():
            samples = self.sampler.get_sample()
            current_states = [a['current_state'] for a in samples]
            next_states = [a['next_state'] for a in samples]
            actions = [a['action'] for a in samples]
            rewards = [a['reward'] for a in samples]
            dones = [1 if a['done'] else 0 for a in samples]
            #@ersin - unutma, asagidaki satir  cartpole icin eklenmis bir kod sadece
            #dones = [1 if a['reward']<0 else 0 for a in samples]
            est_next_states = self.model.predict_next(current_states)
            #print(len(est_next_states))
            ac = (len(est_next_states)-1)/2
            for I in range(len(actions)):
                #print(I, actions[I], next_states[I])
                est_next_states[actions[I]][I] = next_states[I]
                r = min(max(rewards[I], -1), 1)
                est_next_states[ac][I] = r #rewards[I] * self.reward_scale
                est_next_states[ac + 1 + actions[I]][I] = dones[I]
                #est_next_states[actions[I]+ac][I] = 0
            #print(est_next_states)
            loss = self.model.train_next(current_states, est_next_states)
            self.sw.add(loss, self.total_step_count)
            
        #if reward < 0:
        #	print('onstep', ob, action, next_ob, reward, done)
        pass

def output_of_lambda(input_shape):
    return (input_shape[0], 1)

def my_max(x):
    return K.max(x, axis=-1, keepdims=True)
def my_argmax(x):
    return K.cast(K.argmax(x, axis=-1),'float32')

def get_grad(x):
    return K.log(1+K.abs(K.sum(K.gradients(x[0], x[1])[0], axis=1)))
    #return K.sum(x[1], axis=1)

class VNetwork(object):
    def __init__(self, ops):
        self.ops = ops
        self.model = self.get_model()
    def get_model(self):
        raise NotImplementedException()
    def v_value(self, x):
        return self.model.predict_on_batch(x)
    def get_weights(self):
        return self.model.get_weights()
    def set_weights(self, w):
        self.model.set_weights(w)
    def clone(self):
        raise NotImplementedException()
        
class CartPoleVNetwork(VNetwork):
    def __init__(self, ops):
        super(CartPoleVNetwork, self).__init__(ops)
    def get_model(self):
        input_shape=self.ops.INPUT_SIZE
        input = Input(shape=input_shape, name='observation')
        x = input
        x = Dense(24,activation="relu")(x) #, kernel_initializer='he_uniform'
        x = Dense(24,activation="relu")(x) #, kernel_initializer='he_uniform'
        x = Dense(24,activation="relu")(x) #, kernel_initializer='he_uniform'
        v = Dense(1)(x) #activation="relu", , kernel_initializer='he_uniform'
        model = Model(inputs=[input], outputs=[v])
        my_optimizer = RMSprop(lr=self.ops.LEARNING_RATE)
        model.compile(optimizer=my_optimizer,loss='mse')
        return model
    def clone(self):
        new_model = CartPoleVNetwork(self.ops)
        new_model.set_weights(self.get_weights())
        return new_model
    
class MountainCarVNetwork(VNetwork):
    def __init__(self, ops):
        super(MountainCarVNetwork, self).__init__(ops)
    def get_model(self):
        input_shape=self.ops.INPUT_SIZE
        input = Input(shape=input_shape, name='observation')
        x = input
        x = Dense(512,activation="relu")(x) #, kernel_initializer='he_uniform'
        x = Dense(512,activation="relu")(x) #, kernel_initializer='he_uniform'
        x = Dense(512,activation="relu")(x) #, kernel_initializer='he_uniform'
        v = Dense(1)(x) #activation="relu", , kernel_initializer='he_uniform'
        model = Model(inputs=[input], outputs=[v])
        my_optimizer = RMSprop(lr=self.ops.LEARNING_RATE)
        model.compile(optimizer=my_optimizer,loss='mse')
        return model
    def clone(self):
        new_model = MountainCarVNetwork(self.ops)
        new_model.set_weights(self.get_weights())
        return new_model	
    
class LineVNetwork(VNetwork):
    def __init__(self, ops):
        super(LineVNetwork, self).__init__(ops)
    def get_model(self):
        input_shape=self.ops.INPUT_SIZE
        input = Input(shape=input_shape, name='observation')
        x = input
        x = Dense(10,activation="relu")(x) #, kernel_initializer='he_uniform'
        x = Dense(10,activation="relu")(x) # second layer to match DQN variant
        v = Dense(1)(x) #activation="relu", , kernel_initializer='he_uniform'
        model = Model(inputs=[input], outputs=[v])
        #my_optimizer = RMSprop(lr=self.ops.LEARNING_RATE)
        #my_optimizer = Adam(lr=self.ops.LEARNING_RATE)
        #model.compile(optimizer=my_optimizer,loss='mean_absolute_error')
        #self.GRADIENT_MOMENTUM=0.95
        #self.SQUARED_GRADIENT_MOMENTUM=0.95
        #self.MIN_SQUARED_GRADIENT=0.01
        my_optimizer = DqnRMSprop(lr=self.ops.LEARNING_RATE, rho1=0.95, rho2=0.95, epsilon=0.01, print_layer=-1)
        model.compile(optimizer=my_optimizer,loss=huber_loss) #
        return model
    def clone(self):
        new_model = LineVNetwork(self.ops)
        new_model.set_weights(self.get_weights())
        return new_model	
    
class FreewayVNetwork(VNetwork):
    def __init__(self, ops):
        super(FreewayVNetwork, self).__init__(ops)
    def get_model(self):
        input_shape=self.ops.INPUT_SIZE
        if type(input_shape)!=list:
            input_shape = [input_shape]
    
        inputs = []
        flattened_input = []
        for idx,shape in enumerate(input_shape):
            input = Input(shape=shape, name=f'observation{idx}')
            inputs.append(input)
            flattened_input.append(Flatten()(input))
            
        flattened_input = Concatenate()(flattened_input)
    
        x = flattened_input
        x = Dense(256,activation="relu")(x) #, kernel_initializer='he_uniform'
        x = Dense(256,activation="relu")(x) #, kernel_initializer='he_uniform'
        x = Dense(256,activation="relu")(x) #, kernel_initializer='he_uniform'
        #x = Dense(512,activation="relu")(x) #, kernel_initializer='he_uniform'
        v = Dense(1)(x) #activation="relu", , kernel_initializer='he_uniform'
        model = Model(inputs=inputs, outputs=[v])
        my_optimizer = RMSprop(lr=self.ops.LEARNING_RATE)
        model.compile(optimizer=my_optimizer,loss='mse')
        return model
    def clone(self):
        new_model = FreewayVNetwork(self.ops)
        new_model.set_weights(self.get_weights())
        return new_model	    
    
class TDNetwork(object):
    def __init__(self, env_model, v_model, env_ops, include_best_action=False, derivative_coef = 0):
        self.include_best_action = include_best_action 
        self.env_model = env_model
        self.ops = env_ops
        self.v_model = v_model
        self.v_model_eval = v_model.clone()
        self.include_derivative = derivative_coef>0
        self.derivative_coef = derivative_coef
        self.td_model = self.get_model()
    def get_model(self):
        print("TDNETWORK_GETMODEL")
        debug = False
        self.env_model.trainable = False
        input_shape=self.ops.INPUT_SIZE
        if type(input_shape)!=list:
            input_shape = [input_shape]
        inputs = []
        for idx,shape in enumerate(input_shape):
            inputs.append(Input(shape=shape, name=f'observation{idx}'))
        state_length = len(inputs)
        #x = input
        #x = Dense(24,activation="relu")(x) #, kernel_initializer='he_uniform'
        #x = Dense(24,activation="relu")(x) #, kernel_initializer='he_uniform'
        #v = Dense(1)(x) #activation="relu", , kernel_initializer='he_uniform'
        #v_model = Model(inputs=[input], outputs=[v])
        v = self.v_model.model(inputs)
        if debug:
            v = printLayer(v, message='v')
        print("ENV_MODEL_INPUTS", [a.shape for a in inputs])
        print("EXPECTED_INPUTS", [a.shape for a in self.env_model.inputs])
        env_output = self.env_model(inputs)
        #if debug:
        #	for I in range(0, len(env_output)):
        #		env_output[I] = printLayer(env_output[I], message='env_output[{}]'.format(I))
        
        next_v = []
        for I in range(self.ops.ACTION_COUNT):
            #@ersin - use different eval network as in DQN
            #one_v = self.v_model_eval.model(env_output[I])
            #one_v.trainable = False
            one_v = self.v_model.model(env_output[I*state_length:(I+1)*state_length])
            next_v.append(one_v)
        next_v_tensor = Concatenate()(next_v)
        #next_v_tensor = Dropout(0.5)(next_v_tensor)
        #next_v_tensor = GaussianNoise(0.2)(next_v_tensor)
        if debug:
            next_v_tensor = printLayer(next_v_tensor, message='next_v_tensor')
        #@ersin - test icin cikartildi - geri ekle
        next_v_discounted_tensor = Lambda(lambda x: x * 0.99)(next_v_tensor)
        if debug:
            next_v_discounted_tensor = printLayer(next_v_discounted_tensor, message='next_v_discounted_tensor')
        #next_v_discounted_tensor = Lambda(lambda x: x * 0)(next_v_tensor)
        done_tensor = Concatenate()(env_output[self.ops.ACTION_COUNT*state_length+1:])
        if debug:
            done_tensor = printLayer(done_tensor, message='done_tensor')
        #@ersin - test icin cikartildi - geri ekle
        not_done_tensor = Lambda(lambda x: 1 - x)(done_tensor)
        if debug:
            not_done_tensor = printLayer(not_done_tensor, message='not_done_tensor')
        #not_done_tensor = Lambda(lambda x: x*0)(done_tensor)
        done_fix = Multiply()([next_v_discounted_tensor, not_done_tensor])
        if debug:
            done_fix = printLayer(done_fix, message='done_fix')
        reward = env_output[self.ops.ACTION_COUNT*state_length]
        if debug:
            reward = printLayer(reward, message='reward')
        #@ersin - test icin eklendi - cikart
        #reward = Lambda(lambda x: x*0)(reward)
        #@ersin - *100 u yine CartPole icin eklenmisim sanirim
        #reward = Lambda(lambda x: K.switch(x < 0, x*100, x))(reward)
        est_v = Add()([reward, done_fix])
        if debug:
            est_v = printLayer(est_v, message='est_v')
        est_max_v = Lambda(my_max, output_shape=output_of_lambda,name='est_v')(est_v)
        est_best_action = Lambda(my_argmax, output_shape=output_of_lambda,name='best_act')(est_v)
        if debug:
            est_max_v = printLayer(est_max_v, message='est_max_v')
        # disable learning of value func within max
        #est_max_v.trainable = False
        #@ersin - test icin cikartildi - geri ekle
        td_error = Subtract()([v, est_max_v])
        print(inputs, td_error)
        
        if self.include_derivative:
            grad_wrt_input = Lambda(get_grad, output_shape=output_of_lambda, name='grad_diff')([td_error, inputs])
            print(grad_wrt_input)

        if debug:
            td_error = printLayer(td_error, message='td_error')
        #td_error = v

        if self.include_best_action:
            td_model = Model(inputs=inputs, outputs=[td_error, est_best_action])
        elif self.include_derivative:
            td_model = Model(inputs=inputs, outputs=[td_error, grad_wrt_input])
        else:
            td_model = Model(inputs=inputs, outputs=[td_error])
        #my_optimizer = Adam(lr=self.ops.LEARNING_RATE)
        my_optimizer = RMSprop(lr=self.ops.LEARNING_RATE)
        #my_optimizer = Nadam(lr=self.ops.LEARNING_RATE)
        #my_optimizer = Adam(lr=self.ops.LEARNING_RATE,amsgrad=False)
        loss_weights = [1]
        if self.include_derivative or self.include_best_action:
            loss_weights.append(self.derivative_coef)
        td_model.compile(optimizer=my_optimizer,loss='mse', loss_weights=loss_weights)
        #td_model.compile(optimizer=my_optimizer,loss=huber_loss_mse)
        return td_model
    def train(self, state):
        if isinstance(state,list):
            N = len(state[0])
        else:
            N = len(state)
        
        if self.include_derivative:
            return self.td_model.train_on_batch(state, [np.zeros((N, 1), dtype='f'), np.zeros((N, 1), dtype='f')])
        else:
            return self.td_model.train_on_batch(state, np.zeros((N, 1), dtype='f'))
    def test(self, state):
        return self.td_model.predict_on_batch(state)

        
        
