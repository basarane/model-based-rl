import argparse

parser = argparse.ArgumentParser(description='DQN Training')
parser.add_argument('game', type=str, default='Breakout', help='Gym game name')
parser.add_argument('--double-dqn', type=bool, default=False, help='Use double dqn')
parser.add_argument('--dueling-dqn', type=bool, default=False, help='Dueling dqn')
parser.add_argument('--logdir', type=str, default=None, help='Logdir for tensorboard')
parser.add_argument('--enable-render', type=bool, default=False, help='Enable render')
parser.add_argument('--render-step', type=int, default=4, help='render step')

args = parser.parse_args()

from PIL import Image

from envs.gym_env import gym_env
from envs.env_transform import WarmUp, ActionRepeat, ObservationStack, EnvTransform
from utils.preprocess import *
from utils.network_utils import NetworkSaver
from runner.runner import Runner
from agents.agent import DqnAgent, DqnAgentOps, EGreedyOps, EGreedyAgent, MultiEGreedyAgent, EGreedyAgentExp
from utils.memory import ReplayBuffer, NStepBuffer
from nets.net import QModel, DQNModel, DqnOps, init_nn_library
import tensorflow as tf
import keras.backend as K

init_nn_library(True, "1")

from utils.viewer import EnvViewer

class Penalizer(EnvTransform):
	def __init__(self, env):
		super(Penalizer, self).__init__(env)
	def reset(self):
		ob = self.env.reset()
		self.score = 0
		return ob
	def step(self, action):
		ob, reward, done = self.env.step(action)
		reward = reward if not done or self.score == 499 else -100
		self.score += reward
		return ob, reward, done
		
env = gym_env(args.game) # + 'NoFrameskip-v0'
env = Penalizer(env)
#env = WarmUp(env, min_step=0, max_step=30)
#env = ActionRepeat(env, 4)
modelOps = DqnOps(env.action_count)
modelOps.dueling_network = args.dueling_dqn
modelOps.INPUT_SIZE = env.observation_space.shape
modelOps.LEARNING_RATE = 0.001

viewer = None
if args.enable_render:
	viewer = EnvViewer(env, args.render_step, 'human')


proproc = None #PreProPipeline([GrayPrePro(), ResizePrePro(modelOps.INPUT_SIZE)])
rewproc = None #PreProPipeline([RewardClipper(-1, 1)])

from nets.initializers import dqn_uniform
from keras.layers import Input, Permute, ZeroPadding2D, Conv2D, Flatten, Dense, Add, Subtract, Lambda
from keras import Model
from keras.optimizers import RMSprop, Adam
from nets.optimizers import DqnRMSprop
from nets.loss import huber_loss
import numpy as np
import keras.backend as K

class CartPoleModel(QModel):
	def __init__(self, ops = None, model = None):
		super(CartPoleModel, self).__init__(ops, model)
		self.model_update = self.model
	def get_model(self):
		print('*************GET MODEL DQN ***********************')
		input_shape=self.ops.INPUT_SIZE
		input = Input(shape=input_shape, name='observation')
		x = input
		x = Dense(24,activation="relu", kernel_initializer='he_uniform')(x)
		if not self.ops.dueling_network:
			x = Dense(24,activation="relu", kernel_initializer='he_uniform')(x)
			y = Dense(self.ops.ACTION_COUNT, kernel_initializer='he_uniform')(x)
		else:
			xv = Dense(24,activation="relu", kernel_initializer='he_uniform', name="dense_v")(x)
			xa = Dense(24,activation="relu", kernel_initializer='he_uniform', name="dense_a")(x)
			v = Dense(1, kernel_initializer='he_uniform', name="v")(xv) #,activation="relu"
			a = Dense(self.ops.ACTION_COUNT, kernel_initializer='he_uniform', name="a")(xa) #,activation="relu"
			ma = Lambda(my_mean, arguments={'ACTION_COUNT': self.ops.ACTION_COUNT}, name="mean_a")(a)
			y1 = Add(name="v_plus_a")([v, a])
			y = Subtract(name="q_value")([y1, ma])
		model = Model(inputs=[input], outputs=[y])
		#model.summary()
		#model.compile(optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),loss=huber_loss)
		#my_optimizer = DqnRMSprop(lr=self.ops.LEARNING_RATE, rho1=self.ops.GRADIENT_MOMENTUM, rho2=self.ops.SQUARED_GRADIENT_MOMENTUM, epsilon=self.ops.MIN_SQUARED_GRADIENT, print_layer=-1)
		my_optimizer = Adam(lr=self.ops.LEARNING_RATE)
		model.compile(optimizer=my_optimizer,loss='mse') #
		#model.compile(optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),loss='mse')
		return model
	def q_value(self, state):
		state = np.array(state, dtype='f')
		return self.model.predict_on_batch(state)
	def q_update(self, state, target):
		state = np.array(state, dtype='f')
		loss = self.model_update.train_on_batch(state, target)
		if self.model_update is not self.model:
			self.model.set_weights(self.model_update.get_weights())
		return loss
	def get_weights(self):
		return self.model.get_weights()
	def set_weights(self, w):
		self.model.set_weights(w)
	def clone(self):
		m = self.clone_model()
		return CartPoleModel(self.ops, m)

q_model = CartPoleModel(modelOps)

summary_writer = tf.summary.FileWriter(args.logdir, K.get_session().graph) if not args.logdir is None else None

agentOps = DqnAgentOps()
agentOps.double_dqn = args.double_dqn
agentOps.TARGET_NETWORK_UPDATE_FREQUENCY = 20
#agentOps.REPLAY_START_SIZE = 100
#agentOps.FINAL_EXPLORATION_FRAME = 10000

replay_buffer = ReplayBuffer(int(2000), 1, 1, 1000, 64)
#replay_buffer = NStepBuffer(modelOps.AGENT_HISTORY_LENGTH, 8)
agent = DqnAgent(env.action_space, q_model, replay_buffer, rewproc, agentOps, summary_writer)

egreedyOps = EGreedyOps()
egreedyOps.REPLAY_START_SIZE = replay_buffer.REPLAY_START_SIZE
egreedyOps.FINAL_EXPLORATION_FRAME = 10000
egreedyOps.FINAL_EXPLORATION = 0.01
egreedyOps.DECAY = 0.999
egreedyAgent = EGreedyAgentExp(env.action_space, egreedyOps, agent)

runner = Runner(env, egreedyAgent, proproc, 1)
runner.listen(replay_buffer, proproc)
runner.listen(agent, None)
runner.listen(egreedyAgent, None)
if viewer is not None:
	runner.listen(viewer, None)

if args.logdir is not None:
	networkSaver = NetworkSaver(50000, args.logdir, q_model.model)
	runner.listen(networkSaver, None)

runner.run()
