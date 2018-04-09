
class Env(object):
	def __init__(self):
		pass
	
	@property
	def action_space(self):
		raise NotImplementedException()
	
	@property
	def action_count(self):
		pass
		
	def reset(self):
		raise NotImplementedException()
	
	def step(self, action):
		raise NotImplementedException()
