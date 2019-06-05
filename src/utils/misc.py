
#ParameterDecay('linear', 1, 0.1, 20000)
#ParameterDecay('expo', 1, 0.1, 20000)

from ast import literal_eval

def my_literal_eval(param):
	if isinstance(param,str):
		return literal_eval(param)
	else:
		return param

class ParameterDecay:
	def __init__(self, param, dtype='float', default=None):
		self.param = my_literal_eval(param)
		self.dtype = dtype
		self.static_val = None
		if not isinstance(self.param, tuple):
			self.static_val = self.param
			if self.static_val is None:
				self.static_val = default
		else:
			self.dynamic_val = self.param[1]
			if self.param[0] != 'linear' and self.param[0] != 'expo':
				raise Exception('first parameter must be linear or expo')
			if len(self.param)!=4:
				raise Exception('4 parameters needed')
			if self.param[0] == 'expo':
				self.decay = pow(1.0*self.param[2]/self.param[1],1.0/self.param[3])
			elif self.param[0] == 'linear':
				self.decay = 1.0*(self.param[2]-self.param[1])/self.param[3]
		self.step_count = 0
	def on_step(self,**args):
		self.step_count += 1
		if self.static_val is None:
			if self.param[0] == 'linear':
				self.dynamic_val += self.decay
			elif self.param[0] == 'expo':
				self.dynamic_val *= self.decay
			if self.param[2]<self.param[1]:
				self.dynamic_val = max(self.dynamic_val, self.param[2])
			else:
				self.dynamic_val = min(self.dynamic_val, self.param[2])
	def __call__(self):
		if self.static_val is not None:
			return self.static_val
		else:
			if self.dtype == 'int':
				return int(round(self.dynamic_val))
			else:
				return self.dynamic_val
	def is_step(self):
		if self.step_count % self() == 0:
			self.step_count = 0
			return True
		return False
		#a b c
		#
		#
		#a*x^c = b
		#x^c = b/a
		#c log x = log b/a
		#logx = (log b/a)/c
		#x = exp (log b/a)*(1/c)
		#x = (b/a)^(1/c)
		#(a-b)/c
		
print(__name__)
if __name__ == "__main__":
	#par = ParameterDecay("('linear', 1, 0.1, 100)")
	par = ParameterDecay("('linear', 1, 1000, 100000)", 'int')
	#par = ParameterDecay(7)
	last = 0
	for I in range(par.param[3]+20):
		#print(I,par())
		par.on_step()
		if par.is_step():
			print(I-last, I, par())
			last = I
