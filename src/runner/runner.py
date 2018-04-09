
class RunnerListener(object):
	def on_step(self, ob, action, next_ob, reward, done):
		pass
	def on_episode_start(self):
		pass
	def on_episode_end(self, episode_reward, step_count):
		pass
	
class Runner(object):
	def __init__(self, env, agent):
		self.env = env
		self.agent = agent
		self.episode_reward = 0
		self.step_count = 0
		self.total_step_count = 0
		self.listeners = []
		
	def run(self):
		self.total_step_count = 0
		while True:
			ob = self.env.reset()
			[a['listener'].on_episode_start() for a in self.listeners]
			self.episode_reward = 0
			self.step_count = 0
			while True:
				action = self.agent.act(ob)
				next_ob, reward, done = self.env.step(action)
				# preprocess - once per preprocessor
				ob_procs = {}
				next_ob_procs = {}
				for a in self.listeners:
					ob_proc = ob
					next_ob_proc = next_ob
					if a['preproc'] is not None:
						if not id(a['preproc']) in ob_procs:
							ob_procs[id(a['preproc'])] = a['preproc'].preprocess(ob)
							next_ob_procs[id(a['preproc'])] = a['preproc'].preprocess(next_ob)
						ob_proc = ob_procs[id(a['preproc'])]
						next_ob_proc = next_ob_procs[id(a['preproc'])] 
					a['listener'].on_step(ob_proc, action, next_ob_proc, reward, done)
				# notify observers
				#[a['listener'].on_step(ob_proc, action, next_ob_proc, reward, done) for a in self.listeners]
				self.episode_reward = self.episode_reward + reward
				self.total_step_count = self.total_step_count + 1
				self.step_count = self.step_count + 1
				if done:
					[a['listener'].on_episode_end(self.episode_reward, self.step_count) for a in self.listeners]
					break
	def listen(self, obj, preproc = None):
		self.listeners.append({'listener': obj, 'preproc': preproc})
