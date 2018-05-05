from runner.runner import RunnerListener

#		from gym.envs.classic_control import rendering
#		self.viewer = rendering.SimpleImageViewer()
#		self.viewer.imshow(ob)

class EnvViewer(RunnerListener):
	def __init__(self, env, render_step = 4, mode='human'):
		self.env = env
		self.render_step = render_step
		self.total_step = 0
		self.mode = mode
	def render(self):
		self.env.render(self.mode)
	def on_step(self, ob, action, next_ob, reward, done):
		self.total_step	+= 1
		if self.total_step % self.render_step == 0:
			self.render()
