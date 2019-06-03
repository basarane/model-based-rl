import random

class RingBuffer(object):
	def __init__(self,max_size=100):
		self.max_size = max_size
		self.reset()
	def __len__(self):
		return self.length
	def __getitem__(self, idx):
		if isinstance(idx, (list,)):
			return [self[a] for a in idx]
		if idx<0 or idx>=self.length:
			raise KeyError()
		return self.items[idx]
	def append(self, item):
		if self.length < self.max_size:
			self.length = self.length+1
		self.items[self.current] = item
		self.current = (self.current+1)%self.max_size
	def reset(self):
		self.current = 0
		self.length = 0
		self.items = [None for x in range(self.max_size)]
	def __str__(self):
		return str(self.items[0:100])

class SequentialMemory(object):
	def __init__(self, max_size=100):
		self.buffer = RingBuffer(max_size=max_size)
	def append(self, current_state, action, next_state, reward, done):
		self.buffer.append({
			'current_state': current_state,
			'action': action,
			'next_state': next_state,
			'reward': reward,
			'done': done
		})
	def getItems(self, idx, count):
		done = False
		checkCount = 0
		while not done:
			items = self.getItemsInternal(idx, count)
			done = len([x for x in items[:-1] if x['done']]) == 0
			pos1 = idx-count-1
			pos2 = idx
			if (self.buffer.current>pos1 and self.buffer.current<=pos2) or (self.buffer.current>pos1+len(self.buffer) and self.buffer.current<=pos2+len(self.buffer)):
				#print('Whops', idx, count, pos1, pos2, done, self.buffer.current)
				done = False
			checkCount = checkCount+1
			if checkCount%1000000 == 0:
				print('check', checkCount,idx,count,len(self.buffer))
			idx = idx-1
		return items
	def getItemsInternal(self, idx, count):
		return [self.buffer[(idx-I+len(self.buffer))%len(self.buffer)] for I in range(count-1,-1,-1)]
	def getLastItems(self, count):
		items = self.getItems(self.buffer.current-1, count)
		return items
	def samples(self, MINIBATCH_SIZE, skewed_sampling = False):
		samples = random.sample(range(len(self)), MINIBATCH_SIZE)
		if skewed_sampling:
			by_reward = self.group_by_reward()
			per_reward = len(by_reward) / MINIBATCH_SIZE
			samples = []
			for idxes in by_reward:
				this_sample = random.sample(idxes, min(per_reward, len(idxes)))
				samples.extend(this_sample)
			if len(samples) > MINIBATCH_SIZE:
				samples = samples[0:MINIBATCH_SIZE]
			elif len(samples) < MINIBATCH_SIZE:
				samples.extend(random.sample(range(len(self)), MINIBATCH_SIZE-len(samples)))
		return samples
	def group_by_reward(self):
		rewards = set(map(lambda x:None if x is None else x['reward'], self.buffer.items))
		by_reward = [[idx for idx,y in enumerate(self.buffer.items) if not y is None and y['reward']==x] for x in rewards if not x is None]
		return by_reward

	def __getitem__(self, idx):
		return self.buffer[idx]
	def __len__(self):
		return len(self.buffer)
	def reset(self):
		self.buffer.reset()
		
from runner.runner import RunnerListener

class BaseSampler(object):
	def get_sample(self):
		raise NotImplementedException()
	def has_sample(self):
		raise NotImplementedException()
	def nstep(self):
		raise NotImplementedException()

class ReplayBuffer(RunnerListener, BaseSampler):
	def __init__(self, REPLAY_MEMORY_SIZE, AGENT_HISTORY_LENGTH, UPDATE_FREQUENCY, REPLAY_START_SIZE, MINIBATCH_SIZE):
		self.buffer = SequentialMemory(max_size=REPLAY_MEMORY_SIZE)
		self.AGENT_HISTORY_LENGTH = AGENT_HISTORY_LENGTH
		self.UPDATE_FREQUENCY = UPDATE_FREQUENCY
		self.REPLAY_START_SIZE = REPLAY_START_SIZE
		self.MINIBATCH_SIZE = MINIBATCH_SIZE
		self.total_step_count = 0
	def on_step(self, ob, action, next_ob, reward, done):
		self.buffer.append(ob, action, next_ob, reward, done)
		self.total_step_count += 1
	def get_items_internal(self, samples):
		a = []
		for s in samples:
			x = self.buffer.getItems(s, self.AGENT_HISTORY_LENGTH)
			next_state = [b['current_state'] for b in x[1:]]
			next_state.append(x[-1]['next_state'])
			obj = {
				'current_state': [b['current_state'] for b in x],
				'action': x[-1]['action'],
				'next_state': next_state,
				'reward': x[-1]['reward'],
				'done': x[-1]['done']
			}
			# if history_length is 1, return raw state instead of list
			if len(obj['current_state']) == 1:
				obj['current_state'] = obj['current_state'][0]
				obj['next_state'] = obj['next_state'][0]
			a.append(obj)
		return a
	def get_sample(self, skewed_sampling = False):
		samples = self.buffer.samples(self.MINIBATCH_SIZE, skewed_sampling)
		return self.get_items_internal(samples)
	def has_sample(self):
		return self.total_step_count % self.UPDATE_FREQUENCY == 0 and self.total_step_count>self.REPLAY_START_SIZE	
	def nstep(self):
		return False
	
class NStepBuffer(ReplayBuffer):
	def __init__(self, AGENT_HISTORY_LENGTH, N):
		self.buffer = SequentialMemory(max_size=N+AGENT_HISTORY_LENGTH+5)
		self.AGENT_HISTORY_LENGTH = AGENT_HISTORY_LENGTH
		self.N = N
		self.REPLAY_START_SIZE = 1
	def on_step(self, ob, action, next_ob, reward, done):
		self.buffer.append(ob, action, next_ob, reward, done)
	def get_items_internal(self, samples):
		a = []
		for s in samples:
			x = self.buffer.getItemsInternal(s, self.AGENT_HISTORY_LENGTH)
			next_state = [b['current_state'] for b in x[1:]]
			next_state.append(x[-1]['next_state'])
			obj = {
				'current_state': [b['current_state'] for b in x],
				'action': x[-1]['action'],
				'next_state': next_state,
				'reward': x[-1]['reward'],
				'done': x[-1]['done']
			}
			if len(obj['current_state']) == 1:
				obj['current_state'] = obj['current_state'][0]
				obj['next_state'] = obj['next_state'][0]
			a.append(obj)
		return a
	def get_sample(self):
		samples = range(self.AGENT_HISTORY_LENGTH-1, len(self.buffer))
		items = self.get_items_internal(samples)
		if self.buffer[len(self.buffer)-1]['done']:
			self.buffer.reset()
		else:
			last_transitions = [self.buffer[a] for a in range(len(self.buffer)-self.AGENT_HISTORY_LENGTH+1, len(self.buffer))]
			self.buffer.reset()
			for a in last_transitions:
				self.buffer.buffer.append(a)
		return items
		#return self.buffer[samples]
	def has_sample(self):
		return (len(self.buffer)-self.AGENT_HISTORY_LENGTH+1 >= self.N or self.buffer[len(self.buffer)-1]['done']) and len(self.buffer)>=self.AGENT_HISTORY_LENGTH
	def nstep(self):
		return True

if __name__ == "__main__":
	#buffer = RingBuffer(1000000)
	#buffer.append(1)
	#print(buffer)
	#buffer.append(2)
	#print(buffer[1])
	#print(buffer)
	#buffer.append(3)
	#print(buffer)
	#buffer.append(4)
	#print(buffer)
	#buffer.append(5)
	#print(buffer)
	#buffer.append(6)
	#print(buffer)	
	m = SequentialMemory(max_size=100)
	import numpy as np
	counts = np.zeros(40, dtype="int")
	for J in range(10):
		for I in range(40):
			m.append(I, J, I+1, 0, I==39 and J<9)
	#print([a['current_state'] + 1000*a['action'] for a in m.buffer.items])
			if len(m)>32:
				s = m.samples(32)
				for a in s:
					x = m.getItems(a, 4)
					d = [a['current_state'] for a in x]
					counts[d[0]] += 1
					#counts[a['current_state']] += 1
					b = [a['action'] for a in x]
					found = False
					first = b[0]
					for c in b:
						if c!=first:
							found = True
							print(found, b)
							break
					#print([a['current_state'] + 1000*a['action'] for a in x])
				
	print(counts)