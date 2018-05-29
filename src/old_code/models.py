import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Permute, Activation, concatenate, Input, Reshape, Conv2DTranspose, ZeroPadding2D, Cropping2D, Multiply, Layer, Lambda
from keras.initializers import RandomUniform
from keras.objectives import mean_squared_error
from keras.utils import to_categorical
import keras.backend as K

def get_dqn_model(AGENT_HISTORY_LENGTH, INPUT_SIZE, ACTION_COUNT, LEARNING_RATE, GRADIENT_MOMENTUM, SQUARED_GRADIENT_MOMENTUM, MIN_SQUARED_GRADIENT):
	model = Sequential()
	input_shape=(AGENT_HISTORY_LENGTH,) + INPUT_SIZE
	model.add(Permute((2,3,1), input_shape=input_shape))
	model.add(Conv2D(filters=32,kernel_size=8,strides=4,padding="valid",activation="relu"))
	model.add(Conv2D(filters=64,kernel_size=4,strides=2,padding="valid",activation="relu"))
	model.add(Conv2D(filters=64,kernel_size=3,strides=1,padding="valid",activation="relu"))
	model.add(Flatten())
	model.add(Dense(512,activation="relu"))
	model.add(Dense(ACTION_COUNT))
	model.compile(optimizer=keras.optimizers.Nadam(lr=LEARNING_RATE, beta_1=GRADIENT_MOMENTUM, beta_2=SQUARED_GRADIENT_MOMENTUM, epsilon=MIN_SQUARED_GRADIENT),loss='mse')
	return model


def get_acvp_encoder(AGENT_HISTORY_LENGTH):
	model = Sequential(name='f_R')
	input_shape=(AGENT_HISTORY_LENGTH*3,) + (210,160)
	model.add(Permute((2,3,1), input_shape=input_shape))
	model.add(ZeroPadding2D(padding=(0, 1)))
	model.add(Conv2D(filters=64,kernel_size=8,strides=2,padding="valid",activation="relu",kernel_initializer='glorot_normal'))
	model.add(ZeroPadding2D(padding=(1, 1)))
	model.add(Conv2D(filters=128,kernel_size=6,strides=2,padding="valid",activation="relu",kernel_initializer='glorot_normal'))
	model.add(ZeroPadding2D(padding=(1, 1)))
	model.add(Conv2D(filters=128,kernel_size=6,strides=2,padding="valid",activation="relu",kernel_initializer='glorot_normal'))
	model.add(Conv2D(filters=128,kernel_size=4,strides=2,padding="valid",activation="relu",kernel_initializer='glorot_normal'))
	model.add(Flatten())
	model.add(Dense(2048,activation="relu",kernel_initializer='glorot_normal'))
	model.add(Dense(2048,activation="relu",kernel_initializer=RandomUniform(-1, 1))) #'glorot_normal'
	return model

def get_acvp_decoder():
	model = Sequential(name='f_O')
	model.add(Dense(11*8*128,activation="relu",kernel_initializer='glorot_normal', input_shape=(2048,)))
	model.add(Reshape((11,8,128)))
	model.add(Conv2DTranspose(filters=128, kernel_size=4, strides=2, activation='relu', padding='valid',kernel_initializer='glorot_normal'))
	model.add(Conv2DTranspose(filters=128, kernel_size=6, strides=2, activation='relu', padding='valid',kernel_initializer='glorot_normal'))
	model.add(Cropping2D(cropping=(1, 1)))
	model.add(Conv2DTranspose(filters=128, kernel_size=6, strides=2, activation='relu', padding='valid',kernel_initializer='glorot_normal'))
	model.add(Cropping2D(cropping=(1, 1)))
	model.add(Conv2DTranspose(filters=3, kernel_size=8, strides=2, padding='valid',kernel_initializer='glorot_normal'))
	model.add(Cropping2D(cropping=(0, 1)))
	return model

def get_fM(STATE_SIZE, ACTION_COUNT, AGENT_HISTORY_LENGTH):
	state = Input(shape=(STATE_SIZE,), name='current_state')
	actions = Input(shape=(ACTION_COUNT*AGENT_HISTORY_LENGTH,), name='action_performed')
	x = Dense(2048,kernel_initializer=RandomUniform(-0.1, 0.1))(actions)
	x = Multiply()([x, state])
	next_state = Dense(2048, name='next_state')(x)
	model = Model(inputs=[state,actions], outputs=[next_state], name='f_M') #
	return model

def get_fI(STATE_SIZE, ACTION_COUNT):
	model = Sequential(name='f_I')
	model.add(Dense(2048,activation="relu", input_shape=(STATE_SIZE*2,),kernel_initializer=RandomUniform(-0.1, 0.1)))
	model.add(Dense(ACTION_COUNT))
	return model

def get_acvp(STATE_SIZE, ACTION_COUNT, AGENT_HISTORY_LENGTH):
	observations = Input(shape=((AGENT_HISTORY_LENGTH*3,) + (210,160)), name="observation")
	encoder = get_acvp_encoder(AGENT_HISTORY_LENGTH)
	decoder = get_acvp_decoder()
	f_M = get_fM(STATE_SIZE, ACTION_COUNT, AGENT_HISTORY_LENGTH)
	encoded = encoder(observations)
	actions = Input(shape=((ACTION_COUNT*AGENT_HISTORY_LENGTH,)))
	#x = Dense(2048,kernel_initializer=RandomUniform(-0.1, 0.1))(actions)
	#x = Multiply()([x, encoded])
	#x = Dense(2048)(x)
	#predicted = decoder(x)
	next_state = f_M([encoded, actions])
	predicted = decoder(next_state)
	model = Model(inputs=[observations,actions], outputs=[predicted]) #
	model_state = Model(inputs=[observations], outputs=[encoded])

	model.compile(optimizer=keras.optimizers.Nadam(lr=0.0001),loss='mse')
	return model, model_state

def zero_loss(y_true, y_pred):
#	return K.zeros_like(y_pred)
	return y_pred

class CustomMSERegularizer(Layer):
	def __init__(self, coef, coef_var, **kwargs):
		super(CustomMSERegularizer, self).__init__(**kwargs)
		self.coef = coef
		self.coef_var = coef_var
	def call(self, x, mask = None):
		mse2 = mean_squared_error(x[0], x[1])
		loss2 = K.sum(mse2)*self.coef_var*self.coef
		self.add_loss(loss2, x)
		return loss2
	def get_output_shape_for(self, input_shape):
		#return (1,)
		return (input_shape[0][0],1)
	def compute_output_shape(self, input_shape):
		return (input_shape[0][0],1)

class MyCallback(keras.callbacks.Callback):
	def on_batch_end(self, batch, logs={}):
		print('batch_end', logs.get('loss'))
#myCallback = MyCallback()	

def get_env(STATE_SIZE, ACTION_COUNT, AGENT_HISTORY_LENGTH, lossa, var_loss_coef1, var_loss_coef2, var_loss_coef3):
	observations = Input(shape=((AGENT_HISTORY_LENGTH*3,) + (210,160)), name="observation")
	next_observations = Input(shape=((AGENT_HISTORY_LENGTH*3,) + (210,160)), name="next_observation")
	encoder = get_acvp_encoder(AGENT_HISTORY_LENGTH)
	decoder = get_acvp_decoder()
	f_M = get_fM(STATE_SIZE, ACTION_COUNT, AGENT_HISTORY_LENGTH)
	f_I = get_fI(STATE_SIZE, ACTION_COUNT)
	state = encoder(observations)
	actions = Input(shape=((ACTION_COUNT*AGENT_HISTORY_LENGTH,)),name='action')
	last_action = Lambda(lambda x: x[:,ACTION_COUNT*(AGENT_HISTORY_LENGTH-1):], name='last_action')(actions)
	predicted_next_state = f_M([state, actions])
	predicted_next_obs = decoder(predicted_next_state)
	next_state = encoder(next_observations)
	auto_next_obs = decoder(next_state)
	auto_current_obs = decoder(state)
	predicted_actions = f_I(concatenate([state, next_state], name='both_states'))
	reg1 = CustomMSERegularizer(1, var_loss_coef1)([next_state,predicted_next_state])
	reg2 = CustomMSERegularizer(1, var_loss_coef2)([auto_next_obs,predicted_next_obs])
	reg3 = CustomMSERegularizer(1, var_loss_coef3)([last_action, predicted_actions])
	model = Model(inputs=[observations,next_observations,actions], outputs=[predicted_next_obs, auto_current_obs, reg1, reg2, reg3]) #
	model_state = Model(inputs=[observations], outputs=[state])
	model_next_state = Model(inputs=[observations,actions], outputs=[predicted_next_state])
	model_next_state_auto = Model(inputs=[next_observations], outputs=[next_state])
	model.compile(optimizer=keras.optimizers.Nadam(lr=0.0001),loss=['mse', 'mse', zero_loss, zero_loss, zero_loss], loss_weights=[1., lossa, 0., 0., 0.])
	#['loss', 'sequential_3_loss', 'sequential_3_loss', 'custom_mse_regularizer_1_loss', 'custom_mse_regularizer_2_loss', 'custom_mse_regularizer_3_loss']
#	observations  ------------------> state ---------------------> auto_current_obs
#										|
#							  (actions) v predicted_actions
#										|
#										v
#	next_observations --->	(next_state, predicted_next_state) --> (predicted_next_obs, auto_next_obs)
	return model, model_state, model_next_state, model_next_state_auto

def get_reward_estimator(STATE_SIZE, ACTION_COUNT):
	state = Input(shape=((STATE_SIZE,)), name="state")
	action = Input(shape=((ACTION_COUNT,)), name="action")
	state_action = concatenate([state, action], name='state_action')
	#x = Dense(10,activation="relu")(state_action)
	reward = Dense(1)(state_action)
	model = Model(inputs=[state, action], outputs=[reward])
	model.compile(optimizer=keras.optimizers.Nadam(lr=0.0001),loss='mse')
	return model

def load_model(model_name, weight_file, reward_weight, STATE_SIZE, ACTION_COUNT, AGENT_HISTORY_LENGTH, lossa, var_loss_coef1, var_loss_coef2, var_loss_coef3):
	model_env = None
	model_state = None
	model_next_state = None
	model_next_state_auto = None
	model_reward = None
	if model_name == "env":
		model_env, model_state, model_next_state, model_next_state_auto = get_env(STATE_SIZE, ACTION_COUNT, AGENT_HISTORY_LENGTH, lossa, var_loss_coef1, var_loss_coef2, var_loss_coef3)
	else:
		model_env, model_state = get_acvp(STATE_SIZE, ACTION_COUNT, AGENT_HISTORY_LENGTH)

	if len(weight_file)>0:
		model_env.load_weights(weight_file)
	model_reward = get_reward_estimator(STATE_SIZE, ACTION_COUNT)

	if not reward_weight is None and len(reward_weight)>1:
		model_reward.load_weights(reward_weight)
	return model_env, model_state, model_next_state, model_next_state_auto, model_reward

def my_predict(model_env, netin, next_netin, lastActions):
	#pprint(vars(model_env))
	if len(model_env.input_names) == 3:
		prediction,_, _, _, _ = model_env.predict_on_batch([netin, next_netin, lastActions]) #
	else:
		prediction = model_env.predict_on_batch([netin, lastActions]) #
	return prediction