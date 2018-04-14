from PIL import Image

from envs.gym_env import gym_env
from envs.env_transform import WarmUp, ActionRepeat, ObservationStack
from utils.preprocess import *
from runner.runner import Runner
from agents.agent import DqnAgent, DqnAgentOps
from utils.memory import ReplayBuffer
from nets.net import DQNModel, DqnOps, init_nn_library
import tensorflow as tf
import keras.backend as K

init_nn_library(True, "0")

env = gym_env('BreakoutNoFrameskip-v0')
env = WarmUp(env, min_step=0, max_step=30)
env = ActionRepeat(env, 4)
#env = ObservationStack(env, 4)
modelOps = DqnOps(env.action_count)
modelOps.dueling_network = True

proproc = PreProPipeline([GrayPrePro(), ResizePrePro(modelOps.INPUT_SIZE)])
rewproc = PreProPipeline([RewardClipper(-1, 1)])

q_model = DQNModel(modelOps)

summary_writer = tf.summary.FileWriter('test_summary', K.get_session().graph)

agentOps = DqnAgentOps()
agentOps.double_dqn = True
#agentOps.REPLAY_START_SIZE = 100
#agentOps.FINAL_EXPLORATION_FRAME = 5000

replay_buffer = ReplayBuffer(int(1e6))
agent = DqnAgent(env.action_space, q_model, replay_buffer, rewproc, agentOps)

runner = Runner(env, agent, proproc, 4)
runner.listen(replay_buffer, proproc)
runner.listen(agent, proproc)

runner.run()
