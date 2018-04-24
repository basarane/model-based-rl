import argparse

parser = argparse.ArgumentParser(description='DQN Training')
parser.add_argument('game', type=str, default='Breakout', help='Gym game name')
parser.add_argument('--double-dqn', type=bool, default=False, help='Use double dqn')
parser.add_argument('--dueling-dqn', type=bool, default=False, help='Dueling dqn')
parser.add_argument('--logdir', type=str, default=None, help='Logdir for tensorboard')

args = parser.parse_args()

from PIL import Image

from envs.gym_env import gym_env
from envs.env_transform import WarmUp, ActionRepeat, ObservationStack
from utils.preprocess import *
from utils.network_utils import NetworkSaver
from runner.runner import Runner
from agents.agent import DqnAgent, DqnAgentOps
from utils.memory import ReplayBuffer, NStepBuffer
from nets.net import DQNModel, DqnOps, init_nn_library
import tensorflow as tf
import keras.backend as K

init_nn_library(True, "1")

env = gym_env(args.game + 'NoFrameskip-v0')
env = WarmUp(env, min_step=0, max_step=30)
env = ActionRepeat(env, 4)
modelOps = DqnOps(env.action_count)
modelOps.dueling_network = args.dueling_dqn

proproc = PreProPipeline([GrayPrePro(), ResizePrePro(modelOps.INPUT_SIZE)])
rewproc = PreProPipeline([RewardClipper(-1, 1)])

q_model = DQNModel(modelOps)

summary_writer = tf.summary.FileWriter(args.logdir, K.get_session().graph) if not args.logdir is None else None

agentOps = DqnAgentOps()
agentOps.double_dqn = args.double_dqn
#agentOps.REPLAY_START_SIZE = 100
#agentOps.FINAL_EXPLORATION_FRAME = 10000

replay_buffer = ReplayBuffer(int(1e6), 4, 4, agentOps.REPLAY_START_SIZE, 32)
#replay_buffer = NStepBuffer(modelOps.AGENT_HISTORY_LENGTH, 8)
agent = DqnAgent(env.action_space, q_model, replay_buffer, rewproc, agentOps, summary_writer)

runner = Runner(env, agent, proproc, 4)
runner.listen(replay_buffer, proproc)
runner.listen(agent, proproc)

if args.logdir is not None:
	networkSaver = NetworkSaver(50000, args.logdir, q_model.model)
	runner.listen(networkSaver, None)

runner.run()
