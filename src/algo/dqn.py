from PIL import Image

from envs.gym_env import get_env
from envs.gym_env import gym_env
from envs.env_transform import *
from utils.preprocess import *
from utils.network_utils import NetworkSaver
from runner.runner import Runner
from agents.agent import DqnAgent, DqnAgentOps, EGreedyOps, EGreedyAgent, MultiEGreedyAgent, EGreedyAgentExp
from utils.memory import ReplayBuffer, NStepBuffer
from utils.trajectory_utils import TrajectoryReplay
from nets.net import *
import tensorflow as tf
import tensorflow.keras.backend as K
from env_model.model import *


from utils.viewer import EnvViewer

def run_dqn(**kargs):
	if kargs['output_dir'] is None and kargs['logdir'] is not None:
		kargs['output_dir'] = kargs['logdir']

	q_model_initial = kargs['q_model_initial'] if 'q_model_initial' in kargs else None
		
	from collections import namedtuple
	args = namedtuple("DQNParams", kargs.keys())(*kargs.values())

	if 'dont_init_tf' not in kargs.keys() or not kargs['dont_init_tf']:
		#init_nn_library(True, "1")
		init_nn_library("gpu" in kargs and kargs["gpu"] is not None, kargs["gpu"] if "gpu" in kargs else "1")

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
	if 'use_env' in kargs and kargs['use_env'] is not None:
		env = kargs['use_env']
	else:
		env = get_env(args.game, args.atari, args.env_transforms, kargs['monitor_dir'] if 'monitor_dir' in kargs else None)
		if 'env_model' in kargs and kargs['env_model'] is not None and kargs['env_weightfile'] is not None:
			print('Using simulated environment')
			envOps = EnvOps(env.observation_space.shape, env.action_space.n, args.learning_rate)
			env_model = globals()[kargs['env_model']](envOps)
			env_model.model.load_weights(kargs['env_weightfile'])
			env = SimulatedEnv(env, env_model, use_reward='env_reward' in kargs and kargs['env_reward'])

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
	if q_model_initial is None:
		q_model = globals()[args.model](modelOps)
	else:
		q_model = q_model_initial

	if not args.load_weightfile is None:
		q_model.model.load_weights(args.load_weightfile)
	
	summary_writer = tf.summary.FileWriter(args.logdir, K.get_session().graph) if not args.logdir is None else None

	agentOps = DqnAgentOps()
	agentOps.double_dqn = args.double_dqn
	agentOps.mode = args.mode
	if args.mode == "train":
		agentOps.TARGET_NETWORK_UPDATE_FREQUENCY = args.target_network_update

	replay_buffer = None
	if args.replay_buffer_size > 0:
		if 'load_trajectory' in kargs and kargs['load_trajectory'] is not None:
			replay_buffer = TrajectoryReplay(kargs['load_trajectory'], kargs['batch_size'], args.update_frequency, args.replay_start_size)
		else:
			replay_buffer = ReplayBuffer(args.replay_buffer_size, modelOps.AGENT_HISTORY_LENGTH, args.update_frequency, args.replay_start_size, args.batch_size)
	#replay_buffer = NStepBuffer(modelOps.AGENT_HISTORY_LENGTH, 8)
	agent = DqnAgent(env.action_space, q_model, replay_buffer, rewproc, agentOps, summary_writer)

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
			egreedyAgent = EGreedyAgentExp(env.action_space, egreedyOps, agent)
		else:
			egreedyAgent = MultiEGreedyAgent(env.action_space, egreedyOps, agent, args.egreedy_props, args.egreedy_final, final_exp_frame=args.egreedy_final_step)
	else:
		egreedyAgent = EGreedyAgent(env.action_space, egreedyOps, agent)

	runner = Runner(env, egreedyAgent, proproc, modelOps.AGENT_HISTORY_LENGTH, max_step=args.max_step, max_episode=args.max_episode)
	if replay_buffer is not None:
		runner.listen(replay_buffer, proproc)
	runner.listen(agent, None)
	runner.listen(egreedyAgent, None)
	if viewer is not None:
		runner.listen(viewer, None)

	if args.output_dir is not None:
		networkSaver = NetworkSaver(50000 if 'save_interval' not in kargs else kargs['save_interval'], args.output_dir, q_model.model)
		runner.listen(networkSaver, None)

	return runner, agent

def run_dqn_test(**kargs):
	from collections import namedtuple
	args = namedtuple("DQNParams", kargs.keys())(*kargs.values())

	weight_files = []
	if not isinstance(args.load_weightfile,list) or len(args.load_weightfile) == 1:
		wf = args.load_weightfile
		if isinstance(wf,list):
			wf = wf[0]
		weight_files = [(wf,0)]
	else:
		idxs = range(int(args.load_weightfile[1]), int(args.load_weightfile[3]), int(args.load_weightfile[2]))
		weight_files = [(args.load_weightfile[0] + str(I) + '.h5',I) for I in idxs]

	summary_writer = tf.summary.FileWriter(args.logdir, K.get_session().graph) if not args.logdir is None else None		
	sw = SummaryWriter(summary_writer, ['Average reward', 'Total reward'])
	#sw = SummaryWriter(summary_writer, ['Reward'])

	stats = {
		'reward': []
	}
	q_model_initial = None
	last_env = None
	for I, weight_file_info in enumerate(weight_files): 
		weight_file = weight_file_info[0]
		total_step_count = weight_file_info[1]
		arguments = kargs.copy()
		arguments['load_weightfile'] = weight_file
		arguments['q_model_initial'] = q_model_initial
		arguments['use_env'] = last_env
		runner, agent = run_dqn(**arguments)
		last_env = runner.env
		if q_model_initial is None:
			q_model_initial = agent.q_model
		runner.run()
		tmp_stats = np.array(agent.stats['reward'])
		total_reward = tmp_stats[:,1].sum()
		total_reward = total_reward / args.max_episode
		aver_reward = total_reward / tmp_stats[-1,0]
		sw.add([aver_reward, total_reward], I)
		stats['reward'].append((total_step_count, total_reward))
		print('{0} / {1}: Aver Reward per step = {2}, Aver Reward per espisode = {3}'.format(I+1, len(weight_files), aver_reward, total_reward))		
	return stats		
	
	