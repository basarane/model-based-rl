

class Env(object):
	def __init__(self):
		pass
	
	@property
	def action_space(self):
		raise NotImplementedException()
	
	@property
	def observation_space(self):
		raise NotImplementedException()

	@property
	def action_count(self):
		pass
		
	def reset(self):
		raise NotImplementedException()
	
	def step(self, action):
		raise NotImplementedException()

	def render(self, mode='human'):
		raise NotImplementedException()
		
class GridActionSpace(object):
	def __init__(self, n):
		self.n = n
	def sample(self):
		return np.random.randint(self.n)
		
import numpy as np
class GridEnv(object):
	def __init__(self):
		self.x = 0
		self.y = 0
		self.width = 10
		self.height = 10
		self._action_space = GridActionSpace(4)
		self._observation_space = GridActionSpace(self.width*self.height)
	@property
	def action_space(self):
		return self._action_space
	
	@property
	def observation_space(self):
		return self._observation_space

	@property
	def action_count(self):
		return self._action_space.n
		
	def reset(self):
		self.x = 0
		self.y = 0
		return self.get_state()
	def get_state(self):
		return self.x + self.width * self.y
	def step(self, action):
		done = False
		reward = 0
		if action == 0:
			self.x += 1
		if action == 1:
			self.x -= 1
		if action == 2:
			self.y += 1
		if action == 3:
			self.y -= 1
		self.x = max(min(self.x, self.width-1), 0)
		self.y = max(min(self.y, self.height-1), 0)
		s = self.get_state()
		if self.x == self.width-1 and self.y == self.height-1:
			return s, 1, True
		else:
			return s, -0.01, False


import numpy as np
import gym
import random
class LineEnv(object):
	def __init__(self):
		self.x = 0
		self._action_space = gym.spaces.Discrete(2)
		self._observation_space = gym.spaces.Box(-1, 1, (1,))
		self.step_count = 0
	@property
	def action_space(self):
		return self._action_space
	
	@property
	def observation_space(self):
		return self._observation_space

	@property
	def action_count(self):
		return self._action_space.n
		
	def reset(self):
		self.x = random.random()*2-1
		self.step_count = 0
		#return np.zeros((1,))
		return np.array([self.x])
	def step(self, action):
		if action == 0:
			self.x += 0.045 + 0.01*np.random.random()
		if action == 1:
			self.x -= 0.045 + 0.01*np.random.random()
		self.x = max(-1, self.x)
		self.x = min(1, self.x)
		self.step_count += 1
		s = np.zeros((1,))
		s[0] = self.x
		if self.x < 0.55 and self.x > 0.45:
			return s, 1, True
		elif self.step_count < 500:
			return s, -0.01, False
		else:
			return s, -0.01, True

			