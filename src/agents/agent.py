from runner.runner import RunnerListener
import random
import numpy as np

from PIL import Image
import sys

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

class DqnAgentOps(object):
	def __init__(self):
		self.test_epsilon = 0.05
		self.mode = "train"
		self.double_dqn = False
		self.REPLAY_START_SIZE = 50000
		self.INITIAL_EXPLORATION = 1.0
		self.FINAL_EXPLORATION = 0.1
		self.FINAL_EXPLORATION_FRAME = 1e6
		self.UPDATE_FREQUENCY = 4
		self.MINIBATCH_SIZE = 32
		self.DISCOUNT_FACTOR = 0.99
		self.TARGET_NETWORK_UPDATE_FREQUENCY=1e4
		self.AGENT_HISTORY_LENGTH = 4
		
class DqnAgent(Agent, RunnerListener):
	def __init__(self, action_space, q_model, sampler, rewproc, ops):
		super(DqnAgent, self).__init__(action_space, ops)
		print('REPLAY STATER', ops.REPLAY_START_SIZE)
		self.q_model = q_model
		self.q_model_eval = q_model.clone()
		self.sampler = sampler
		self.total_step_count = 0
		self.losses = []
		self.rewproc = rewproc
	def act(self, observation):
		action = self.action_space.sample()
		if (self.total_step_count > self.ops.REPLAY_START_SIZE) or self.ops.mode == "test":
			epsilon = (self.ops.INITIAL_EXPLORATION-self.ops.FINAL_EXPLORATION) * max(self.ops.FINAL_EXPLORATION_FRAME-self.total_step_count, 0) / (self.ops.FINAL_EXPLORATION_FRAME-self.ops.REPLAY_START_SIZE) + self.ops.FINAL_EXPLORATION
			if self.ops.mode == "test":
				epsilon = self.ops.test_epsilon
			if epsilon < random.random():
				prediction = self.q_model.q_value([observation])[0]
				action = np.argmax(prediction)
		return action
		
	def on_step(self, ob, action, next_ob, reward, done):
		self.total_step_count += 1
		#print(self.ops.UPDATE_FREQUENCY, self.total_step_count, self.ops.REPLAY_START_SIZE)
		if self.total_step_count % self.ops.UPDATE_FREQUENCY == 0 and self.total_step_count>self.ops.REPLAY_START_SIZE:
			samples = self.sampler.get_sample(self.ops.MINIBATCH_SIZE, self.ops.AGENT_HISTORY_LENGTH)
			current_states = [a['current_state'] for a in samples]
			next_states = [a['next_state'] for a in samples]
			
			target = self.q_model.q_value(current_states)
			next_value = self.q_model_eval.q_value(next_states)

			if self.ops.double_dqn:
				next_best_res = self.q_model.q_value(next_states)
				best_acts = np.argmax(next_best_res, axis=1)
			else:
				best_acts = np.argmax(next_value, axis=1)
			
			for I in xrange(self.ops.MINIBATCH_SIZE):
				transition = samples[I]
				action = transition['action']
				reward = transition['reward']
				if self.rewproc is not None:
					reward = self.rewproc.preprocess(reward)
				if transition['done']:
					target[I,action] = reward
				else:
					target[I,action] = reward + self.ops.DISCOUNT_FACTOR * next_value[I,best_acts[I]]   #after double DQN
			res = self.q_model.q_update(current_states, target)
			self.losses.append(res)
		if self.total_step_count % self.ops.TARGET_NETWORK_UPDATE_FREQUENCY == 1 and self.ops.mode == "train":
			self.q_model_eval.set_weights(self.q_model.get_weights())
	def on_episode_end(self, reward, step_count):
		x = np.array(self.losses)
		print('Episode end', reward, step_count, self.total_step_count, x.sum() / step_count)
		self.losses = []
