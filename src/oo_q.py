import argparse

parser = argparse.ArgumentParser(description='DQN Training')
parser.add_argument('game', type=str, default='Breakout', help='Gym game name')
parser.add_argument('--double-dqn', type=bool, default=False, help='Use double dqn')
parser.add_argument('--dueling-dqn', type=bool, default=False, help='Dueling dqn')
parser.add_argument('--logdir', type=str, default=None, help='Logdir for tensorboard')
parser.add_argument('--nstep', type=int, default=5, help='step count for n-step q learning')

args = parser.parse_args()

from PIL import Image

from envs.gym_env import gym_env
from envs.env_transform import WarmUp, ActionRepeat, ObservationStack
from utils.preprocess import *
from runner.runner import Runner
from agents.agent import DqnAgent, DqnAgentOps
from utils.memory import ReplayBuffer, NStepBuffer
from nets.net import TabularQModel, DqnOps, init_nn_library
import tensorflow as tf
import keras.backend as K
from envs.env import GridEnv

init_nn_library(True, "0")

if args.game == "Grid":
	env = GridEnv()
else:
	env = gym_env(args.game)


#print(env.observation_space.n)

modelOps = DqnOps(env.action_count)
modelOps.dueling_network = args.dueling_dqn
modelOps.INPUT_SIZE = env.observation_space.n
modelOps.LEARNING_RATE = 0.2

q_model = TabularQModel(modelOps)

summary_writer = tf.summary.FileWriter(args.logdir, K.get_session().graph) if not args.logdir is None else None

agentOps = DqnAgentOps()
agentOps.double_dqn = args.double_dqn
agentOps.REPLAY_START_SIZE = 1
agentOps.FINAL_EXPLORATION_FRAME = 10000

replay_buffer = NStepBuffer(1, args.nstep)
agent = DqnAgent(env.action_space, q_model, replay_buffer, None, agentOps, summary_writer)

runner = Runner(env, agent, None, 1)
runner.listen(replay_buffer, None)
runner.listen(agent, None)

runner.run()
