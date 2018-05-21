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
parser.add_argument('--env-model', type=str, default=None, help='class name of environment model')
parser.add_argument('--learning-rate', type=float, default=0.00025, help='learning rate')
parser.add_argument('--env-transforms', type=str, nargs='*', default=[], help='apply the environment transforms')
parser.add_argument('--update-frequency', type=int, default=4, help='training update frequency')
parser.add_argument('--replay-buffer-size', type=int, default=int(1e6), help='the number of transitions in replay buffer')
parser.add_argument('--replay-start-size', type=int, default=int(50000), help='replay start size')
parser.add_argument('--batch-size', type=int, default=int(32), help='batch size')
parser.add_argument('--max-step', type=int, default=int(1e10), help='max step')
parser.add_argument('--reward-scale', type=float, default=float(0.01), help='reward scale')
parser.add_argument('--save-interval', type=int, default=10000, help='save interval')
parser.add_argument('--load-trajectory', type=str, default=None, help='the trajectory h5 file')

args = parser.parse_args()

from env_model.model import *
from envs.gym_env import get_env
from algo.dqn import run_dqn
from utils.memory import ReplayBuffer
import keras.backend as K
import tensorflow as tf 
from utils.network_utils import NetworkSaver
from runner.runner import TrajRunner
from utils.trajectory_utils import TrajectoryReplay
from nets.net import init_nn_library

arguments = vars(args)

env = get_env(args.game, args.atari, args.env_transforms)

if args.load_trajectory is None:
	dqn_args = arguments.copy()
	dqn_args['mode'] = 'test'
	dqn_args['replay_buffer_size'] = 0

	runner = run_dqn(**dqn_args)
	replay_buffer = ReplayBuffer(args.replay_buffer_size, 1, args.update_frequency, args.replay_start_size, args.batch_size)
else:
	init_nn_library(True, "1")
	runner = TrajRunner(args.max_step)
	replay_buffer = TrajectoryReplay(args.load_trajectory, args.batch_size)

envOps = EnvOps(env.observation_space.shape, env.action_space.n, args.learning_rate)
summary_writer = tf.summary.FileWriter(args.logdir, K.get_session().graph) if not args.logdir is None else None

#model = EnvModelCartpole(envOps)
model = globals()[args.env_model](envOps)
env = EnvLearner(replay_buffer, model, summary_writer, args.reward_scale)

runner.listen(env, None)

if args.output_dir is None:
	args.output_dir = args.logdir
if args.output_dir is not None:
	networkSaver = NetworkSaver(args.save_interval, args.output_dir, model.model)
	runner.listen(networkSaver, None)

runner.run()
