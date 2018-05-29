from envs.gym_env import get_env
from env_model.model import *
import keras.backend as K
import tensorflow as tf 
from nets.net import init_nn_library
from utils.memory import ReplayBuffer
from agents.agent import VAgent, EGreedyOps, EGreedyAgent, MultiEGreedyAgent, EGreedyAgentExp
from td import run_td_test

def run_td_realtime(**kargs):
	if kargs['output_dir'] is None and kargs['logdir'] is not None:
		kargs['output_dir'] = kargs['logdir']

	from collections import namedtuple
	args = namedtuple("TDRealtimeParams", kargs.keys())(*kargs.values())

	if 'dont_init_tf' not in kargs.keys() or not kargs['dont_init_tf']:
		init_nn_library(True, "1")

	env = get_env(args.game, args.atari, args.env_transforms)

	envOps = EnvOps(env.observation_space.shape, env.action_space.n, args.learning_rate, mode="train")
	print(env.observation_space.low)
	print(env.observation_space.high)

	env_model = globals()[args.env_model](envOps)
	if args.env_weightfile is not None:
		env_model.model.load_weights(args.env_weightfile)

	v_model = globals()[args.vmodel](envOps)

	import numpy as np
	td_model = TDNetwork(env_model.model, v_model, envOps)

	summary_writer = tf.summary.FileWriter(args.logdir, K.get_session().graph) if not args.logdir is None else None

	replay_buffer = ReplayBuffer(args.replay_buffer_size, 1, args.update_frequency, args.replay_start_size, args.batch_size)

	from utils.network_utils import NetworkSaver
	network_saver = NetworkSaver(args.save_freq, args.logdir, v_model.model)

	v_agent = VAgent(env.action_space, env_model, v_model, envOps, summary_writer, True, replay_buffer, args.target_network_update)

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
	#runner.run()
	return runner, v_agent

def run_td_realtime_test(**kargs):
	return run_td_test(**kargs)
	