
from env import Env
import numpy as np

class EnvTransform(Env):
	def __init__(self, env):
		super(Env, self).__init__()
		self.env = env
	
	@property
	def action_space(self):
		return self.env.action_space
	
	@property
	def observation_space(self):
		return self.env.observation_space

	@property
	def action_count(self):
		return self.env.action_count
		
	def reset(self):
		return self.env.reset()
	
	def step(self, action):
		return self.env.step(action)
		
	def render(self, mode='human'):
		return self.env.render(mode)

class WarmUp(EnvTransform):
	def __init__(self, env, min_step=0, max_step=30):
		super(WarmUp, self).__init__(env)
		self.min_step = min_step
		self.max_step = max_step
		
	def reset(self):
		ob = self.env.reset()
		noop_count = np.random.random_integers(self.min_step, self.max_step)
		for I in range(noop_count):
			ob, _, done = self.env.step(0)
			if done:
				print('Game terminated during warm up')
		return ob
	
class ActionRepeat(EnvTransform):
	def __init__(self, env, action_repeat=4):
		super(ActionRepeat, self).__init__(env)
		self.action_repeat = action_repeat
		
	def step(self, action):
		tot_reward = 0
		obs = []
		for I in range(self.action_repeat):
			ob, reward, done = self.env.step(action)
			obs.append(ob)
			tot_reward += reward
			if done:
				break
		obs = obs[-2:]
		ob = np.array(obs)
		ob = np.max(ob, 0)
		return ob, tot_reward, done

class ObservationStack(EnvTransform):
	def __init__(self, env, step_count=4):
		super(ObservationStack, self).__init__(env)
		self.step_count = step_count
		self.obs = []
	def reset(self):
		ob = self.env.reset()
		obs = [ob]
		for I in range(self.step_count-1):
			ob, _, done = self.env.step(0)
			obs.append(ob)
			if done:
				print('Game terminated during warm up')
		self.obs = obs
		return obs
	def step(self, action):
		ob, r, done = self.env.step(action)
		self.obs = self.obs[1:]
		self.obs.append(ob)
		return self.obs, r, done
		
class Penalizer(EnvTransform):
	def __init__(self, env):
		super(Penalizer, self).__init__(env)
	def reset(self):
		ob = self.env.reset()
		self.score = 0
		return ob
	def step(self, action):
		ob, reward, done = self.env.step(action)
		reward = reward if not done or self.score == 499 else -5
		self.score += reward
		return ob, reward, done
		
from gym.spaces import Box
class StepEnv(EnvTransform):
	def __init__(self, env):
		super(StepEnv, self).__init__(env)
		self.step_count = 0
	def reset(self):
		ob = self.env.reset()
		ob = self.add_step_count(ob)
		self.step_count = 0
		return ob
	def step(self, action):
		ob, reward, done = self.env.step(action)
		self.step_count += 1
		ob = self.add_step_count(ob)
		return ob, reward, done
	def add_step_count(self, vec):
		vec = np.resize(vec, (vec.shape[0] + 1, ))
		vec[-1] = self.step_count * 0.01
		return vec
	@property
	def observation_space(self):
		shape = (self.env.observation_space.shape[0]+1,)
		x = Box(0, 1, shape)
		return x

class SimulatedEnv(EnvTransform):
	def __init__(self, env, env_model, use_reward = False):
		super(SimulatedEnv, self).__init__(env)
		self.env_model = env_model
		self.last_ob = None
		self.use_reward = use_reward
		print('Use simulated reward: ', use_reward)
	def reset(self):
		self.last_ob = self.env.reset()
		return self.last_ob
	def step(self, action):
		res = self.env_model.predict_next([self.last_ob])
		ob, reward, done = self.env.step(action)
		self.last_ob = ob
		if self.use_reward:
			est_reward = res[self.env.action_count][0][action]
			#print(reward, est_reward)
			#@ersin - burasi cartpole icin eklendi, cikartmayi unutma
			#if est_reward < 0:
			#	est_reward = -100
			reward = est_reward
		#print('ob', ob)
		#print('est', res[action][0])
		return res[action][0], reward, done
