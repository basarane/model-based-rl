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
parser.add_argument('--max-episode', type=int, default=int(1e10), help='max episode')
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
parser.add_argument('--egreedy-props', type=float, nargs='*', default=[1], help='multiple egreedy props')
parser.add_argument('--egreedy-final', type=float, nargs='*', default=[0.1], help='multiple egreedy final exploration')
parser.add_argument('--egreedy-final-step', type=int, nargs='*', default=[int(1e6)], help='multiple egreedy final step')
parser.add_argument('--egreedy-decay', type=float, default=1, help='exponential decay rate for egreedy')

args = parser.parse_args()

from envs.gym_env import get_env
from env_model.model import *
import keras.backend as K
import tensorflow as tf 
from nets.net import init_nn_library
from utils.memory import ReplayBuffer
from agents.agent import VAgent, EGreedyOps, EGreedyAgent, MultiEGreedyAgent, EGreedyAgentExp

arguments = vars(args)

init_nn_library(True, "1")

env = get_env(args.game, args.atari, args.env_transforms)

envOps = EnvOps(env.observation_space.shape, env.action_space.n, args.learning_rate, mode="train")
print(env.observation_space.low)
print(env.observation_space.high)

env_model = globals()[args.env_model](envOps)
env_model.model.load_weights(args.env_weightfile)

v_model = globals()[args.vmodel](envOps)

import numpy as np
td_model = TDNetwork(env_model.model, v_model, envOps)

summary_writer = tf.summary.FileWriter(args.logdir, K.get_session().graph) if not args.logdir is None else None

replay_buffer = ReplayBuffer(args.replay_buffer_size, 1, args.update_frequency, args.replay_start_size, args.batch_size)

from utils.network_utils import NetworkSaver
network_saver = NetworkSaver(args.save_freq, args.logdir, v_model.model)

v_agent = VAgent(env.action_space, env_model, v_model, envOps, summary_writer, True, replay_buffer)

egreedyOps = EGreedyOps()
if replay_buffer is not None:
	egreedyOps.REPLAY_START_SIZE = replay_buffer.REPLAY_START_SIZE
egreedyOps.mode = args.mode
egreedyOps.test_epsilon = args.test_epsilon
#egreedyOps.FINAL_EXPLORATION_FRAME = 10000
if args.mode == "train":
	egreedyOps.FINAL_EXPLORATION_FRAME = args.egreedy_final_step

if args.mode == "train":
	if args.egreedy_decay<1:
		egreedyOps.DECAY = args.egreedy_decay
		egreedyAgent = EGreedyAgentExp(env.action_space, egreedyOps, v_agent)
	else:
		egreedyAgent = MultiEGreedyAgent(env.action_space, egreedyOps, v_agent, args.egreedy_props, args.egreedy_final, final_exp_frame=args.egreedy_final_step)
else:
	egreedyAgent = EGreedyAgent(env.action_space, egreedyOps, v_agent)


runner = Runner(env, egreedyAgent, None, 1, max_step=args.max_step, max_episode=args.max_episode)
runner.listen(replay_buffer, None)
runner.listen(v_agent, None)
runner.listen(egreedyAgent, None)
runner.listen(network_saver, None)
runner.run()

