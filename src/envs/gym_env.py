import gym

from env import Env

class gym_env(Env):
	def __init__(self, rom_name):
		super(Env, self).__init__()
		self.env = gym.make(rom_name)
	
	@property
	def action_space(self):
		return self.env.action_space
	
	@property
	def observation_space(self):
		return self.env.observation_space

	@property
	def action_count(self):
		return self.env.action_space.n
		
	def reset(self):
		return self.env.reset()
	
	def step(self, action):
		#print('gym step')
		ob, r, done, _ = self.env.step(action)
		if r>4.9:
			print(r)
		return ob, r, done

	def render(self, mode='human'):
		return self.env.render(mode)
