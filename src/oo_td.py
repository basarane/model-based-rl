import argparse

parser = argparse.ArgumentParser(description='DQN Training')
parser.add_argument('game', type=str, default='Breakout', help='Gym game name')
parser.add_argument('--mode', type=str, default="train", help='mode: train or test')
parser.add_argument('--env-model', type=str, default=None, help='class name of environment model')
parser.add_argument('--env-weightfile', type=str, default=None, help='load environment weights')
parser.add_argument('--output-dir', type=str, default=None, help='output directory')
parser.add_argument('--logdir', type=str, default=None, help='Logdir for tensorboard')
parser.add_argument('--learning-rate', type=float, default=0.00025, help='learning rate')
parser.add_argument('--batch-size', type=int, default=int(32), help='batch size')
parser.add_argument('--max-step', type=int, default=int(1e10), help='max step')
parser.add_argument('--save-interval', type=int, default=10000, help='save interval')
parser.add_argument('--target-network-update', type=int, default=1000, help='target network update feq')
parser.add_argument('--update-frequency', type=int, default=4, help='training update frequency')
parser.add_argument('--replay-buffer-size', type=int, default=int(1e6), help='the number of transitions in replay buffer')
parser.add_argument('--replay-start-size', type=int, default=int(50000), help='replay start size')
parser.add_argument('--test-epsilon', type=float, default=0.05, help='epsilon for testing')
parser.add_argument('--load-weightfile', type=str, default=None, help='load initial weights')
parser.add_argument('--atari', type=bool, default=False, help='true if env is atari game')
parser.add_argument('--env-transforms', type=str, nargs='*', default=[], help='apply the environment transforms')
parser.add_argument('--dueling-dqn', type=bool, default=False, help='Dueling dqn')
parser.add_argument('--enable-render', type=bool, default=False, help='Enable render')
parser.add_argument('--model', type=str, default='DQNModel', help='class name for q-model')
parser.add_argument('--double-dqn', type=bool, default=False, help='Use double dqn')
parser.add_argument('--load-trajectory', type=str, default=None, help='load sample trajectories from this file')
parser.add_argument('--vmodel', type=str, default='V Model', help='class name for v-model')
parser.add_argument('--save-freq', type=int, default=5000, help='save network after this many batches')


args = parser.parse_args()

from envs.gym_env import get_env
from env_model.model import *
import keras.backend as K
import tensorflow as tf 
from nets.net import init_nn_library

arguments = vars(args)

init_nn_library(True, "1")

env = get_env(args.game, args.atari, args.env_transforms)

envOps = EnvOps(env.observation_space.shape, env.action_space.n, args.learning_rate)
print(env.observation_space.low)
print(env.observation_space.high)

env_model = globals()[args.env_model](envOps)
env_model.model.load_weights(args.env_weightfile)

v_model = globals()[args.vmodel](envOps)

import numpy as np
td_model = TDNetwork(env_model.model, v_model, envOps)

summary_writer = tf.summary.FileWriter(args.logdir, K.get_session().graph) if not args.logdir is None else None
sw = SummaryWriter(summary_writer, ['Loss'])


#dqn_args = arguments.copy()
#dqn_args['mode'] = 'test'
#dqn_args['replay_buffer_size'] = 0
#
#class TDListener(RunnerListener):
#	def __init__(self, replay_buffer, td_model):
#		self.replay_buffer = replay_buffer
#		self.td_model = td_model
#		self.I = 0
#	def on_step(self, ob, action, next_ob, reward, done):
#		if self.replay_buffer.has_sample():
#			samples = self.replay_buffer.get_sample()
#			#samples = np.array([a['current_state'] for a in samples], dtype='f')
#			samples = np.random.uniform([-4.8, -5, -0.48, -5], [4.8, 5, 0.48, 5], size=(args.batch_size,4))
#			loss = self.td_model.train(samples)
#			sw.add([loss], I)
#			if self.I % args.target_network_update == 0:
#				self.td_model.v_model_eval.set_weights(self.td_model.v_model.get_weights())
#			self.I += 1
#
#from algo.dqn import run_dqn			
#from utils.memory import ReplayBuffer
#
#runner = run_dqn(**dqn_args)
#replay_buffer = ReplayBuffer(args.replay_buffer_size, 1, args.update_frequency, args.replay_start_size, args.batch_size)
#runner.listen(replay_buffer, None)
#runner.listen(TDListener(replay_buffer, td_model), None)
#runner.run()

from utils.trajectory_utils import TrajectoryLoader
traj = TrajectoryLoader(args.load_trajectory)

from utils.network_utils import NetworkSaver
network_saver = NetworkSaver(args.save_freq, args.logdir, v_model.model)

for I in xrange(args.max_step):
	#batch = np.random.uniform([-4.8, -5, -0.48, -5], [4.8, 5, 0.48, 5], size=(args.batch_size,4))
	batch = traj.sample(args.batch_size)
	loss = td_model.train(batch['current'])
	sw.add([loss], I)
	network_saver.on_step()
	if I % args.target_network_update == 0:
		td_model.v_model_eval.set_weights(td_model.v_model.get_weights())
	