import gym
from gym import wrappers

from env import *
from env_transform import *

def get_env(game, atari = False, env_transforms = [], monitor_dir = None):
	if atari:
		env = gym_env(game + 'NoFrameskip-v0', monitor_dir)
		env = WarmUp(env, min_step=0, max_step=30)
		env = ActionRepeat(env, 4)
	else:
		if game == "Grid":
			env = GridEnv()
		elif game == "Line":
			env = LineEnv()
		else:
			env = gym_env(game, monitor_dir)
	for trans in env_transforms:
		env = globals()[trans](env)
	return env
	
class gym_env(Env):
	def __init__(self, rom_name, monitor_dir = None):
		super(Env, self).__init__()
		self.env = gym.make(rom_name)
		if monitor_dir is not None:
			self.env = wrappers.Monitor(self.env, monitor_dir, force=True, video_callable=lambda episode_id: True)
	
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
