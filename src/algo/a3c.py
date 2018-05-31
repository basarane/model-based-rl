from PIL import Image

from envs.gym_env import gym_env
from envs.env_transform import *
from envs.env import GridEnv
from utils.preprocess import *
from runner.runner import Runner, RunnerListener
from agents.agent import *
from utils.memory import ReplayBuffer, NStepBuffer
from nets.net import *
import tensorflow as tf
import keras.backend as K

import threading
from threading import Lock
from utils.stopable_thread import StoppableThread

def run_a3c(**kargs):
	if kargs['output_dir'] is None and kargs['logdir'] is not None:
		kargs['output_dir'] = kargs['logdir']

	from collections import namedtuple
	args = namedtuple("A3CParams", kargs.keys())(*kargs.values())

	if len(args.egreedy_props)>1 and args.egreedy_props[0] == round(args.egreedy_props[0]):
		if not round(np.array(args.egreedy_props).sum()) == args.thread_count-1:
			print('thread_count '+str(args.thread_count)+' should be one more than the sum of egreedy_props ' +str(np.array(args.egreedy_props).sum()))
			return
	
	ENABLE_RENDER = args.enable_render
	if not args.output_dir is None:
		import os, sys
		if not os.path.exists(args.output_dir):
			os.makedirs(args.output_dir)
			
	if ENABLE_RENDER:
		from gym.envs.classic_control import rendering
		viewer = rendering.SimpleImageViewer()

	init_nn_library(False, "1")

	if args.atari:
		env = gym_env(args.game + 'NoFrameskip-v0')
	else:
		if args.game == "Grid":
			env = GridEnv()
		else:
			env = gym_env(args.game)

	modelOps = DqnOps(env.action_count)
	modelOps.dueling_network = args.dueling_dqn

	if args.atari:
		#model = A3CModel(modelOps)
		SAVE_FREQ = 100000
		#modelOps.LEARNING_RATE = 0.00025
		#target_network_update_freq = 10000
	else:
		if env.observation_space.__class__.__name__ is 'Discrete':
			modelOps.INPUT_SIZE = env.observation_space.n
		else:
			modelOps.INPUT_SIZE = env.observation_space.shape
		modelOps.AGENT_HISTORY_LENGTH = 1
		#modelOps.LEARNING_RATE = 0.2
		#model = TabularQModel(modelOps)
		#target_network_update_freq = 100
		SAVE_FREQ = 1e10
		
	if 'save_interval' in kargs and kargs['save_interval'] is not None:
		SAVE_FREQ = kargs['save_interval']

	modelOps.LEARNING_RATE = args.learning_rate
	target_network_update_freq = args.target_network_update
		
	global T, model_eval, model

	model = globals()[args.model](modelOps)

	T = 0
	tLock = Lock()

	sess = tf.InteractiveSession()
	K.set_session(sess) 

	model_eval = model.clone()

	sess.run(tf.global_variables_initializer())

	graph = tf.get_default_graph()

	class AgentThread(StoppableThread, RunnerListener):
		def __init__(self, threadId, sess, graph):
			StoppableThread.__init__(self)
			self.threadId = threadId
			self.sess = sess
			self.graph = graph
			with self.graph.as_default():
				if args.atari:
					env = gym_env(args.game + 'NoFrameskip-v0')
					env = WarmUp(env, min_step=0, max_step=30)
					env = ActionRepeat(env, 4)
					proproc = PreProPipeline([GrayPrePro(), ResizePrePro(modelOps.INPUT_SIZE)])
					rewproc = PreProPipeline([RewardClipper(-1, 1)])
					#q_model = A3CModel(modelOps)
				else:
					if args.game == "Grid":
						env = GridEnv()
					else:
						env = gym_env(args.game)
					proproc = None
					rewproc = None
					#q_model = TabularQModel(modelOps)
				for trans in args.env_transforms:
					env = globals()[trans](env)
				q_model = globals()[args.model](modelOps)

				q_model.model_update = model.model
				q_model.set_weights(model.get_weights())
				summary_writer = tf.summary.FileWriter(args.logdir + '/thread-'+str(threadId), K.get_session().graph) if not args.logdir is None else None

				agentOps = DqnAgentOps()
				agentOps.double_dqn = args.double_dqn
				agentOps.REPLAY_START_SIZE = 1
				#agentOps.INITIAL_EXPLORATION = 0
				agentOps.TARGET_NETWORK_UPDATE_FREQUENCY = 1e10

				#replay_buffer = ReplayBuffer(int(1e6), 4, 4, agentOps.REPLAY_START_SIZE, 32)
				replay_buffer = None
				#if threadId > 0:
				if args.nstep > 0:
					replay_buffer = NStepBuffer(modelOps.AGENT_HISTORY_LENGTH, args.nstep)
				else:
					replay_buffer = ReplayBuffer(args.replay_buffer_size, modelOps.AGENT_HISTORY_LENGTH, args.update_frequency, args.replay_start_size, args.batch_size)

				agent = DqnAgent(env.action_space, q_model, replay_buffer, rewproc, agentOps, summary_writer, model_eval=model_eval) #

				egreedyAgent = None
				
				if threadId > 0: # first thread is for testing
					egreedyOps = EGreedyOps()
					egreedyOps.REPLAY_START_SIZE = replay_buffer.REPLAY_START_SIZE
					#egreedyOps.FINAL_EXPLORATION_FRAME = int(args.egreedy_final_step / args.thread_count)				
					#if args.egreedy_decay<1:
					#	egreedyAgent = EGreedyAgentExp(env.action_space, egreedyOps, agent)
					#else:
					if len(args.egreedy_props)>1 and args.egreedy_props[0] == round(args.egreedy_props[0]):
						cs = np.array(args.egreedy_props)
						cs = np.cumsum(cs)
						idx = np.searchsorted(cs, threadId)
						print('Egreedyagent selected', idx, args.egreedy_final[idx], args.egreedy_decay[idx], args.egreedy_final_step[idx])
						egreedyAgent = MultiEGreedyAgent(env.action_space, egreedyOps, agent, [1], [args.egreedy_final[idx]], [args.egreedy_decay[idx]], [args.egreedy_final_step[idx]])
					else:
						egreedyAgent = MultiEGreedyAgent(env.action_space, egreedyOps, agent, args.egreedy_props, args.egreedy_final, args.egreedy_decay, args.egreedy_final_step)
				
				self.runner = Runner(env, egreedyAgent if egreedyAgent is not None else agent, proproc, modelOps.AGENT_HISTORY_LENGTH)
				if replay_buffer is not None:
					self.runner.listen(replay_buffer, proproc)
				self.runner.listen(agent, None)
				if egreedyAgent is not None:
					self.runner.listen(egreedyAgent, None)
				self.runner.listen(self, proproc)
				self.agent = agent
				self.q_model = q_model
			pass
		def run(self):
			with self.graph.as_default():
				self.runner.run()
		def on_step(self, ob, action, next_ob, reward, done):
			global T
			global model, model_eval
			with tLock:
				T = T + 1
				if T % target_network_update_freq == 0:
					#print('CLONE TARGET: ' + str(T))
					model_eval.set_weights(model.get_weights())
					for agent in agents:
						agent.model_eval = model_eval
				if T % SAVE_FREQ == 0 and args.mode == "train":
					if not args.output_dir is None:
						model.model.save_weights(args.output_dir + '/weights_{0}.h5'.format(T))
			#if T % 1000 == 0:
			#	print('STEP', T)
			#if self.threadId == 0 and T % 10 == 0:
			#	self.q_model.set_weights(model.get_weights())
			if T%args.render_step == 0 and ENABLE_RENDER:
				viewer.imshow(np.repeat(np.reshape(ob, ob.shape + (1,)), 3, axis=2))
			if T > args.max_step:
				self.stop()
			#print(T)
		def stop(self):
			super(AgentThread, self).stop()
			self.runner.stop()

	import time
	agents = []
	for I in range(args.thread_count):
		agent = AgentThread(I, sess, graph)
		agents.append(agent)

	print("NOW START THE AGENTS ****************")
	for agent in agents:
		time.sleep(1)
		agent.start()
		#self.default_graph.finalize()
		
	while True:
		try:
			#print('HEART BEAT')
			time.sleep(1)
			stopped = True
			for agent in agents:
				stopped = stopped and agent.stopped()
			if stopped:
				break
		except KeyboardInterrupt: 
			print('KILL AGENTS')
			for agent in agents:
				agent.stop()
			print('TIME TO STOP')
			sess.close()
			return None
	print("All threads are stopped: ", T)
	sess.close()
	return agents[0].agent.stats