
from runner.runner import RunnerListener

class NetworkSaver(RunnerListener):
	def __init__(self, SAVE_FREQ, output_dir, model):
		self.total_step_count = 0
		self.SAVE_FREQ = SAVE_FREQ
		self.output_dir = output_dir
		self.model = model
	def on_step(self, ob = None, action = None, next_ob = None, reward = None, done = None):
		self.total_step_count += 1
		if self.total_step_count % self.SAVE_FREQ == 0:
			if not self.output_dir is None:
				self.model.save_weights(self.output_dir + '/weights_{0}.h5'.format(self.total_step_count))
