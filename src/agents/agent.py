from runner.runner import RunnerListener
import random
import numpy as np
import copy

from PIL import Image
import sys

from utils.summary_writer import SummaryWriter

class Agent(object):
	def __init__(self, action_space, ops):
		self.action_space = action_space
		self.ops = ops
	def act(self, observation):
		raise NotImplementedException()

class RandomAgent(Agent):
	def __init__(self, action_space, ops):
		super(RandomAgent, self).__init__(action_space, ops)
	def act(self, observation):
		return self.action_space.sample()

class EGreedyOps(object):
	def __init__(self):
		self.test_epsilon = 0.05
		self.mode = "train"
		self.REPLAY_START_SIZE = 50000
		self.INITIAL_EXPLORATION = 1.0
		self.FINAL_EXPLORATION = 0.1
		self.FINAL_EXPLORATION_FRAME = 1e6
		
class EGreedyAgent(Agent,RunnerListener):
	def __init__(self, action_space, ops, agent):
		super(EGreedyAgent, self).__init__(action_space, ops)
		self.total_step_count = 0
		self.random_agent = RandomAgent(action_space, ops)
		self.agent = agent
	def act(self, observation):
		action = self.random_agent.act(observation)
		if (self.total_step_count > self.ops.REPLAY_START_SIZE) or self.ops.mode == "test":
			epsilon = (self.ops.INITIAL_EXPLORATION-self.ops.FINAL_EXPLORATION) * max(self.ops.FINAL_EXPLORATION_FRAME-self.total_step_count, 0) / (self.ops.FINAL_EXPLORATION_FRAME-self.ops.REPLAY_START_SIZE) + self.ops.FINAL_EXPLORATION
			if self.ops.mode == "test":
				epsilon = self.ops.test_epsilon
			if epsilon < random.random():
				action = self.agent.act(observation)
		return action
	def on_step(self, ob, action, next_ob, reward, done):
		self.total_step_count += 1
		
class MultiEGreedyAgent(Agent,RunnerListener):
	def __init__(self, action_space, ops, agent, dist_prop=[1], final_exp=[0.1]):
		super(MultiEGreedyAgent, self).__init__(action_space, ops)
		self.dist_prop = dist_prop
		self.greedy_agents = []
		for x in final_exp:
			tmp_ops = copy.copy(ops)
			tmp_ops.FINAL_EXPLORATION = x
			tmp_agent = EGreedyAgent(action_space, tmp_ops, agent)
			self.greedy_agents.append(tmp_agent)
	def act(self, observation):
		agent = np.random.choice(self.greedy_agents, 1, self.dist_prop)
		return agent[0].act(observation)
		
class DqnAgentOps(object):
	def __init__(self):
		self.test_epsilon = 0.05
		self.mode = "train"
		self.double_dqn = False
		self.MINIBATCH_SIZE = 32
		self.DISCOUNT_FACTOR = 0.99
		self.TARGET_NETWORK_UPDATE_FREQUENCY=1e4
		
class DqnAgent(Agent, RunnerListener):
	def __init__(self, action_space, q_model, sampler, rewproc, ops, sw = None, model_eval=None):
		super(DqnAgent, self).__init__(action_space, ops)
		self.q_model = q_model
		if model_eval is not None:
			self.q_model_eval = model_eval
		else:
			self.q_model_eval = q_model.clone()
		self.sampler = sampler
		self.total_step_count = 0
		self.losses = []
		self.rewproc = rewproc
		self.sw = SummaryWriter(sw, ['Episode reward', 'Loss per batch'])
	def act(self, observation):
		prediction = self.q_model.q_value([observation])[0]
		action = np.argmax(prediction)
		return action
		
	def on_step(self, ob, action, next_ob, reward, done):
		self.total_step_count += 1
		if self.sampler.has_sample():
			samples = self.sampler.get_sample()
			current_states = [a['current_state'] for a in samples]
			next_states = [a['next_state'] for a in samples]
			
			target = self.q_model.q_value(current_states)
			next_value = self.q_model_eval.q_value(next_states)

			if self.ops.double_dqn:
				next_best_res = self.q_model.q_value(next_states)
				best_acts = np.argmax(next_best_res, axis=1)
			else:
				best_acts = np.argmax(next_value, axis=1)
			
			R = 0
			for I in reversed(xrange(len(samples))):
				transition = samples[I]
				action = transition['action']
				reward = transition['reward']
				if self.rewproc is not None:
					reward = self.rewproc.preprocess(reward)
				if (self.sampler.nstep() and I==len(samples)-1) or not self.sampler.nstep():
					R = next_value[I,best_acts[I]]
				if transition['done']:
					R = reward
				else:
					R = reward + self.ops.DISCOUNT_FACTOR * R   #after double DQN
				target[I,action] = R
			res = self.q_model.q_update(current_states, target)
			self.losses.append(res)
			self.update_count += 1
		if self.total_step_count % self.ops.TARGET_NETWORK_UPDATE_FREQUENCY == 0 and self.ops.mode == "train":
			self.q_model_eval.set_weights(self.q_model.get_weights())
	def on_episode_start(self):
		self.update_count = 0
	def on_episode_end(self, reward, step_count):
		if len(self.losses)>0:
			x = np.array(self.losses)
			aver_loss = x.sum() / self.update_count
		else:
			aver_loss = 0
		print('Episode end', reward, step_count, self.total_step_count, aver_loss)
		self.update_count = 0
		self.sw.add([reward, aver_loss], self.total_step_count)
		self.losses = []

