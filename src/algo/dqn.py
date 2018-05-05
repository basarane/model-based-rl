from PIL import Image

from envs.gym_env import gym_env
from envs.env_transform import *
from utils.preprocess import *
from utils.network_utils import NetworkSaver
from runner.runner import Runner
from agents.agent import DqnAgent, DqnAgentOps, EGreedyOps, EGreedyAgent, MultiEGreedyAgent, EGreedyAgentExp
from utils.memory import ReplayBuffer, NStepBuffer
from nets.net import *
import tensorflow as tf
import keras.backend as K

from utils.viewer import EnvViewer

def run_dqn(**kargs):
	if kargs['output_dir'] is None and kargs['logdir'] is not None:
		kargs['output_dir'] = kargs['logdir']

	from collections import namedtuple
	args = namedtuple("DQNParams", kargs.keys())(*kargs.values())

	init_nn_library(True, "1")

	if args.atari:
		env = gym_env(args.game + 'NoFrameskip-v0')
		env = WarmUp(env, min_step=0, max_step=30)
		env = ActionRepeat(env, 4)
		#q_model = A3CModel(modelOps)
	else:
		if args.game == "Grid":
			env = GridEnv()
		else:
			env = gym_env(args.game)
		#q_model = TabularQModel(modelOps)
	for trans in args.env_transforms:
		env = globals()[trans](env)

	modelOps = DqnOps(env.action_count)
	modelOps.dueling_network = args.dueling_dqn

	viewer = None
	if args.enable_render:
		viewer = EnvViewer(env, args.render_step, 'human')
	if args.atari:
		proproc = PreProPipeline([GrayPrePro(), ResizePrePro(modelOps.INPUT_SIZE)])
		rewproc = PreProPipeline([RewardClipper(-1, 1)])
	else:
		if env.observation_space.__class__.__name__ is 'Discrete':
			modelOps.INPUT_SIZE = env.observation_space.n
		else:
			modelOps.INPUT_SIZE = env.observation_space.shape
		modelOps.AGENT_HISTORY_LENGTH = 1
		proproc = None
		rewproc = None

	modelOps.LEARNING_RATE = args.learning_rate
	q_model = globals()[args.model](modelOps)

	if not args.load_weightfile is None:
		q_model.model.load_weights(args.load_weightfile)
	
	summary_writer = tf.summary.FileWriter(args.logdir, K.get_session().graph) if not args.logdir is None else None

	agentOps = DqnAgentOps()
	agentOps.double_dqn = args.double_dqn
	agentOps.mode = args.mode
	agentOps.TARGET_NETWORK_UPDATE_FREQUENCY = args.target_network_update

	replay_buffer = ReplayBuffer(args.replay_buffer_size, modelOps.AGENT_HISTORY_LENGTH, args.update_frequency, args.replay_start_size, args.batch_size)
	#replay_buffer = NStepBuffer(modelOps.AGENT_HISTORY_LENGTH, 8)
	agent = DqnAgent(env.action_space, q_model, replay_buffer, rewproc, agentOps, summary_writer)

	egreedyOps = EGreedyOps()
	egreedyOps.REPLAY_START_SIZE = replay_buffer.REPLAY_START_SIZE
	egreedyOps.mode = args.mode
	egreedyOps.test_epsilon = args.test_epsilon
	#egreedyOps.FINAL_EXPLORATION_FRAME = 10000
	egreedyOps.FINAL_EXPLORATION_FRAME = args.egreedy_final_step

	if args.egreedy_decay<1:
		egreedyOps.DECAY = args.egreedy_decay
		egreedyAgent = EGreedyAgentExp(env.action_space, egreedyOps, agent)
	else:
		egreedyAgent = MultiEGreedyAgent(env.action_space, egreedyOps, agent, args.egreedy_props, args.egreedy_final)
	#egreedyAgent = EGreedyAgent(env.action_space, egreedyOps, agent)

	runner = Runner(env, egreedyAgent, proproc, modelOps.AGENT_HISTORY_LENGTH, max_step=args.max_step)
	runner.listen(replay_buffer, proproc)
	runner.listen(agent, None)
	runner.listen(egreedyAgent, None)
	if viewer is not None:
		runner.listen(viewer, None)

	if args.output_dir is not None:
		networkSaver = NetworkSaver(50000, args.output_dir, q_model.model)
		runner.listen(networkSaver, None)

	runner.run()
