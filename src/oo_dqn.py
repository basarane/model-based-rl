import argparse

parser = argparse.ArgumentParser(description='DQN Training')
parser.add_argument('game', type=str, default='Breakout', help='Gym game name')
parser.add_argument('--mode', type=str, default="train", help='mode: train or test')
parser.add_argument('--test-epsilon', type=float, default=0.05, help='epsilon for testing')
parser.add_argument('--load-weightfile', type=str, default=None, help='load initial weights')
parser.add_argument('--output-dir', type=str, default=None, help='output directory')
parser.add_argument('--double-dqn', type=bool, default=False, help='Use double dqn')
parser.add_argument('--dueling-dqn', type=bool, default=False, help='Dueling dqn')
parser.add_argument('--logdir', type=str, default=None, help='Logdir for tensorboard')
parser.add_argument('--enable-render', type=bool, default=False, help='Enable render')
parser.add_argument('--render-step', type=int, default=4, help='render step')
parser.add_argument('--atari', type=bool, default=False, help='true if env is atari game')
parser.add_argument('--model', type=str, default='DQNModel', help='class name for q-model')
parser.add_argument('--learning-rate', type=float, default=0.00025, help='learning rate')
parser.add_argument('--target-network-update', type=int, default=10000, help='target network update feq')
parser.add_argument('--egreedy-props', type=float, nargs='*', default=[1], help='multiple egreedy props')
parser.add_argument('--egreedy-final', type=float, nargs='*', default=[0.1], help='multiple egreedy final exploration')
parser.add_argument('--egreedy-final-step', type=int, nargs='*', default=[int(1e6)], help='multiple egreedy final step')
parser.add_argument('--egreedy-decay', type=float, default=1, help='exponential decay rate for egreedy')
parser.add_argument('--env-transforms', type=str, nargs='*', default=[], help='apply the environment transforms')
parser.add_argument('--update-frequency', type=int, default=4, help='training update frequency')
parser.add_argument('--replay-buffer-size', type=int, default=int(1e6), help='the number of transitions in replay buffer')
parser.add_argument('--replay-start-size', type=int, default=int(50000), help='replay start size')
parser.add_argument('--batch-size', type=int, default=int(32), help='batch size')
parser.add_argument('--max-step', type=int, default=int(1e10), help='max step')
parser.add_argument('--load-trajectory', type=str, default=None, help='load sample trajectories from this file and use as ReplayMemory')
parser.add_argument('--save-trajectory', type=str, default=None, help='save trajectories to this file')
parser.add_argument('--dont-init-tf', type=bool, default=False, help='do not init tensorflow library, do it automatically')
parser.add_argument('--env-model', type=str, default=None, help='use simulated env model')
parser.add_argument('--env-weightfile', type=str, default=None, help='weightfile for simulated env model')
parser.add_argument('--env-reward', type=bool, default=False, help='use the estimated rewards from environment model')
parser.add_argument('--save-interval', type=int, default=50000, help='save interval')

args = parser.parse_args()

from algo.dqn import run_dqn

arguments = vars(args)

import numpy as np


runner, _ = run_dqn(**arguments)
if args.save_trajectory is not None:
	from utils.trajectory_utils import TrajectorySaver
	ts = TrajectorySaver(args.save_trajectory)
	runner.listen(ts, None)
runner.run()

#from PIL import Image
#
#from envs.gym_env import gym_env
#from envs.env_transform import *
#from utils.preprocess import *
#from utils.network_utils import NetworkSaver
#from runner.runner import Runner
#from agents.agent import DqnAgent, DqnAgentOps, EGreedyOps, EGreedyAgent, MultiEGreedyAgent, EGreedyAgentExp
#from utils.memory import ReplayBuffer, NStepBuffer
#from nets.net import *
#import tensorflow as tf
#import keras.backend as K
#
#init_nn_library(True, "1")
#
#from utils.viewer import EnvViewer
#
#if args.atari:
#	env = gym_env(args.game + 'NoFrameskip-v0')
#	env = WarmUp(env, min_step=0, max_step=30)
#	env = ActionRepeat(env, 4)
#	#q_model = A3CModel(modelOps)
#else:
#	if args.game == "Grid":
#		env = GridEnv()
#	else:
#		env = gym_env(args.game)
#	#q_model = TabularQModel(modelOps)
#for trans in args.env_transforms:
#	env = globals()[trans](env)
#
#modelOps = DqnOps(env.action_count)
#modelOps.dueling_network = args.dueling_dqn
#
#viewer = None
#if args.enable_render:
#	viewer = EnvViewer(env, args.render_step, 'human')
#if args.atari:
#	proproc = PreProPipeline([GrayPrePro(), ResizePrePro(modelOps.INPUT_SIZE)])
#	rewproc = PreProPipeline([RewardClipper(-1, 1)])
#else:
#	if env.observation_space.__class__.__name__ is 'Discrete':
#		modelOps.INPUT_SIZE = env.observation_space.n
#	else:
#		modelOps.INPUT_SIZE = env.observation_space.shape
#	modelOps.AGENT_HISTORY_LENGTH = 1
#	proproc = None
#	rewproc = None
#
#modelOps.LEARNING_RATE = args.learning_rate
#q_model = globals()[args.model](modelOps)
#
#summary_writer = tf.summary.FileWriter(args.logdir, K.get_session().graph) if not args.logdir is None else None
#
#agentOps = DqnAgentOps()
#agentOps.double_dqn = args.double_dqn
#agentOps.TARGET_NETWORK_UPDATE_FREQUENCY = args.target_network_update
#
#replay_buffer = ReplayBuffer(args.replay_buffer_size, modelOps.AGENT_HISTORY_LENGTH, args.update_frequency, args.replay_start_size, args.batch_size)
##replay_buffer = NStepBuffer(modelOps.AGENT_HISTORY_LENGTH, 8)
#agent = DqnAgent(env.action_space, q_model, replay_buffer, rewproc, agentOps, summary_writer)
#
#egreedyOps = EGreedyOps()
#egreedyOps.REPLAY_START_SIZE = replay_buffer.REPLAY_START_SIZE
##egreedyOps.FINAL_EXPLORATION_FRAME = 10000
#egreedyOps.FINAL_EXPLORATION_FRAME = args.egreedy_final_step
#
#if args.egreedy_decay<1:
#	egreedyOps.DECAY = args.egreedy_decay
#	egreedyAgent = EGreedyAgentExp(env.action_space, egreedyOps, agent)
#else:
#	egreedyAgent = MultiEGreedyAgent(env.action_space, egreedyOps, agent, args.egreedy_props, args.egreedy_final)
##egreedyAgent = EGreedyAgent(env.action_space, egreedyOps, agent)
#
#runner = Runner(env, egreedyAgent, proproc, modelOps.AGENT_HISTORY_LENGTH)
#runner.listen(replay_buffer, proproc)
#runner.listen(agent, None)
#runner.listen(egreedyAgent, None)
#if viewer is not None:
#	runner.listen(viewer, None)
#
#if args.logdir is not None:
#	networkSaver = NetworkSaver(50000, args.logdir, q_model.model)
#	runner.listen(networkSaver, None)
#
#runner.run()
