import gym

from env import *
from env_transform import *

def get_env(game, atari = False, env_transforms = []):
	if atari:
		env = gym_env(game + 'NoFrameskip-v0')
		env = WarmUp(env, min_step=0, max_step=30)
		env = ActionRepeat(env, 4)
	else:
		if game == "Grid":
			env = GridEnv()
		elif game == "Line":
			env = LineEnv()
		else:
			env = gym_env(game)
	for trans in env_transforms:
		env = globals()[trans](env)
	return env
	
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
		#if r>4.9:
		#	print(r)
		return ob, r, done

	def render(self, mode='human'):
		return self.env.render(mode)
