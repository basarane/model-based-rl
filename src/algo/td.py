from envs.gym_env import get_env
from env_model.model import *
import keras.backend as K
import tensorflow as tf 
from nets.net import init_nn_library
from agents.agent import VAgent

def run_td(**kargs):
	if kargs['output_dir'] is None and kargs['logdir'] is not None:
		kargs['output_dir'] = kargs['logdir']

	from collections import namedtuple
	args = namedtuple("TDParams", kargs.keys())(*kargs.values())

	init_nn_library(True, "1")

	env = get_env(args.game, args.atari, args.env_transforms)

	envOps = EnvOps(env.observation_space.shape, env.action_space.n, args.learning_rate)
	print(env.observation_space.low)
	print(env.observation_space.high)

	env_model = globals()[args.env_model](envOps)
	if args.env_weightfile is not None:
		env_model.model.load_weights(args.env_weightfile)

	v_model = globals()[args.vmodel](envOps)

	import numpy as np
	td_model = TDNetwork(env_model.model, v_model, envOps)

	summary_writer = tf.summary.FileWriter(args.logdir, K.get_session().graph) if not args.logdir is None else None
	sw = SummaryWriter(summary_writer, ['Loss'])

	if args.load_trajectory is not None:
		from utils.trajectory_utils import TrajectoryLoader
		traj = TrajectoryLoader(args.load_trajectory)

	from utils.network_utils import NetworkSaver
	network_saver = NetworkSaver(args.save_freq, args.logdir, v_model.model)

	import scipy.stats as stats

	exponent = 2
	if 'td_exponent' in kargs and kargs['td_exponent'] is not None:
		exponent = kargs['td_exponent']
	
	for I in xrange(args.max_step):
		#batch = np.random.uniform([-4.8, -5, -0.48, -5], [4.8, 5, 0.48, 5], size=(args.batch_size,4))
		#lower, upper = -1, 1
		#mu, sigma = 0.5, 0.4
		#X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
		#samples = np.random.uniform([-1], [1], size=(5000,1))
		samples = np.random.uniform(args.smin, args.smax, size=(args.sample_count,len(args.smin)))
		td_errors = td_model.test(samples).flatten()
		props = np.abs(td_errors)
		#props = np.multiply(props, props)
		props = np.power(props, exponent)
		props = props / props.sum()
		idxs = np.random.choice(samples.shape[0], args.batch_size, True, props)
		batch = {
			#'current': np.random.uniform([-1], [1], size=(args.batch_size,1))
			#'current': X.rvs((args.batch_size,1))
			'current': samples[idxs]
		}
		#batch = traj.sample(args.batch_size)
		loss = td_model.train(batch['current'])
		sw.add([loss], I)
		network_saver.on_step()
		if I % args.target_network_update == 0:
			td_model.v_model_eval.set_weights(td_model.v_model.get_weights())
		

def run_td_test(**kargs):
	if ('output_dir' not in kargs or kargs['output_dir'] is None) and kargs['logdir'] is not None:
		kargs['output_dir'] = kargs['logdir']

	from collections import namedtuple
	args = namedtuple("TDTestParams", kargs.keys())(*kargs.values())

	init_nn_library(True, "1")

	#env = gym_env(args.game)
	env = get_env(args.game, args.atari, args.env_transforms)

	viewer = None
	if args.enable_render:
		viewer = EnvViewer(env, args.render_step, 'human')

	envOps = EnvOps(env.observation_space.shape, env.action_space.n, 0)
	print(env.observation_space.low)
	print(env.observation_space.high)

	env_model = globals()[args.env_model](envOps)
	if args.env_weightfile is not None:
		env_model.model.load_weights(args.env_weightfile)

	v_model = globals()[args.vmodel](envOps)

	weight_files = []
	if len(args.load_weightfile) == 1:
		weight_files = [(args.load_weightfile,0)]
	else:
		idxs = range(int(args.load_weightfile[1]), int(args.load_weightfile[3]), int(args.load_weightfile[2]))
		weight_files = [(args.load_weightfile[0] + str(I) + '.h5',I) for I in idxs]
		
	summary_writer = tf.summary.FileWriter(args.logdir, K.get_session().graph) if not args.logdir is None else None
		
	sw = SummaryWriter(summary_writer, ['Average reward', 'Total reward'])
	#sw = SummaryWriter(summary_writer, ['Reward'])

	stats = {
		'reward': []
	}
	for I, weight_file_info in enumerate(weight_files): 
		weight_file = weight_file_info[0]
		total_step_count = weight_file_info[1]
		v_model.model.load_weights(weight_file)
		v_agent = VAgent(env.action_space, env_model, v_model, envOps, None, False)
		runner = Runner(env, v_agent, None, 1, max_step=args.max_step, max_episode=args.max_episode)
		runner.listen(v_agent, None)
		if viewer is not None:
			runner.listen(viewer, None)
		runner.run()
		tmp_stats = np.array(v_agent.stats['reward'])
		total_reward = tmp_stats[:,1].sum()
		total_reward = total_reward / args.max_episode
		aver_reward = total_reward / tmp_stats[-1,0]
		sw.add([aver_reward, total_reward], I)
		stats['reward'].append((total_step_count, total_reward))
		print('{0} / {1}: Aver Reward per step = {2}, Aver Reward per espisode = {3}'.format(I+1, len(weight_files), aver_reward, total_reward))		
	return stats