
class RunnerListener(object):
	def on_step(self, ob, action, next_ob, reward, done):
		pass
	def on_episode_start(self):
		pass
	def on_episode_end(self, episode_reward, step_count):
		pass

class BaseRunner(object):
	def __init__(self, max_step = None, max_episode = None):
		self.max_step = max_step
		self.max_episode = max_episode
		self.listeners = []
		self.stopped = False
	def stop(self):
		self.stopped = True
	def listen(self, obj, preproc = None):
		self.listeners.append({'listener': obj, 'preproc': preproc})
	def run(self):
		raise NotImplementedException()
		
class Runner(BaseRunner):
	def __init__(self, env, agent, agent_preproc = None, agent_step_count = None, max_step = None, max_episode = None):
		super(Runner, self).__init__(max_step, max_episode)
		self.env = env
		self.agent = agent
		self.episode_reward = 0
		self.step_count = 0
		self.total_step_count = 0
		self.episode_count = 0
		self.agent_preproc = agent_preproc 
		self.agent_step_count = agent_step_count
	def run(self):
		self.total_step_count = 0
		self.episode_count = 0
		while not self.stopped:
			ob = self.env.reset()
			[a['listener'].on_episode_start() for a in self.listeners]
			self.episode_reward = 0
			self.step_count = 0
			ob_procs = {}
			next_ob_procs = {}
			
			obs = []
			while not self.stopped:
				if not self.agent_preproc is None:
					if not id(self.agent_preproc) in ob_procs:
						ob_procs[id(self.agent_preproc)] = self.agent_preproc.preprocess(ob)
					obs.append(ob_procs[id(self.agent_preproc)])
				for a in self.listeners:
					if a['preproc'] is not None:
						if not id(a['preproc']) in ob_procs:
							ob_procs[id(a['preproc'])] = a['preproc'].preprocess(ob)
				if self.agent_step_count > 1 and len(obs) < self.agent_step_count:
					action = self.env.action_space.sample()
				else:
					if self.agent_step_count > 1:
						obs = obs[-self.agent_step_count:]
						action = self.agent.act(obs)
					else:
						action = self.agent.act(ob)
				next_ob, reward, done = self.env.step(action)
				# preprocess - once per preprocessor
				for a in self.listeners:
					ob_proc = ob
					next_ob_proc = next_ob
					if a['preproc'] is not None:
						if not id(a['preproc']) in next_ob_procs:
							next_ob_procs[id(a['preproc'])] = a['preproc'].preprocess(next_ob)
						ob_proc = ob_procs[id(a['preproc'])]
						next_ob_proc = next_ob_procs[id(a['preproc'])] 
					a['listener'].on_step(ob_proc, action, next_ob_proc, reward, done)
				# notify observers
				#[a['listener'].on_step(ob_proc, action, next_ob_proc, reward, done) for a in self.listeners]
				self.episode_reward = self.episode_reward + reward
				self.total_step_count = self.total_step_count + 1
				self.step_count = self.step_count + 1
				ob = next_ob
				ob_procs = next_ob_procs
				next_ob_procs = {}
				if done:
					self.episode_count += 1
					[a['listener'].on_episode_end(self.episode_reward, self.step_count) for a in self.listeners]
					if self.max_episode is not None and self.episode_count >= self.max_episode:
						return
					break
				if self.max_step is not None and self.total_step_count >= self.max_step:
					return

class TrajRunner(BaseRunner):
	def __init__(self, max_step = None):
		super(TrajRunner, self).__init__(max_step)
		self.step = 0
	def run(self):
		self.step = 0
		while True:
			self.step += 1
			for a in self.listeners:
				a['listener'].on_step(None, None, None, None, None)
			if self.max_step is not None and self.step >= self.max_step:
				return
		
