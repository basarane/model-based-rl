
import numpy as np
import h5py
import random

from runner.runner import RunnerListener

class TrajectorySaver(RunnerListener):
	def __init__(self, fname, save_per_step = 100):
		self.fname = fname
		self.step = 0
		self.save_per_step = save_per_step
		self.buffer = None
		self.f = h5py.File(self.fname, "a")
		self.dset = None
	def save(self):
		if self.dset is None:
			if 'current' not in self.f:
				self.dset = {
					'current': self.f.create_dataset('current', (0,self.buffer['current'].shape[1]), maxshape=(None,self.buffer['current'].shape[1]), chunks=(self.save_per_step,self.buffer['current'].shape[1]), dtype=self.buffer['current'].dtype),
					'next': self.f.create_dataset('next', (0,self.buffer['next'].shape[1]), maxshape=(None,self.buffer['next'].shape[1]), chunks=(self.save_per_step,self.buffer['next'].shape[1]), dtype=self.buffer['next'].dtype),
					'action': self.f.create_dataset('action', (0,self.buffer['action'].shape[1]), maxshape=(None,self.buffer['action'].shape[1]), chunks=(self.save_per_step,self.buffer['action'].shape[1]), dtype=self.buffer['action'].dtype),
					'reward': self.f.create_dataset('reward', (0,self.buffer['reward'].shape[1]), maxshape=(None,self.buffer['reward'].shape[1]), chunks=(self.save_per_step,self.buffer['reward'].shape[1]), dtype=self.buffer['reward'].dtype),
					'done': self.f.create_dataset('done', (0,self.buffer['done'].shape[1]), maxshape=(None,self.buffer['done'].shape[1]), chunks=(self.save_per_step,self.buffer['done'].shape[1]), dtype=self.buffer['done'].dtype)
				}
			else:
				self.dset = {
					'current': self.f['current'],
					'next': self.f['next'],
					'reward': self.f['reward'],
					'action': self.f['action'],
					'done': self.f['done']
				}
		for a in self.dset.keys():
			self.dset[a].resize(self.dset[a].shape[0] + self.save_per_step, axis=0)
			self.dset[a][-self.save_per_step:,:] = self.buffer[a]
		self.buffer = None
		self.f.flush()
	def on_step(self, ob, action, next_ob, reward, done):
		if self.buffer is None:
			self.buffer = {
				'current': np.zeros((self.save_per_step,) + ob.shape, dtype = ob.dtype),
				'next': np.zeros((self.save_per_step,) + next_ob.shape, dtype = next_ob.dtype),
				'reward': np.zeros((self.save_per_step,1), dtype = 'float'),
				'action': np.zeros((self.save_per_step,1), dtype = 'int'),
				'done': np.zeros((self.save_per_step,1), dtype = 'bool')
			}
		self.buffer['current'][self.step % self.save_per_step] = ob
		self.buffer['next'][self.step % self.save_per_step] = next_ob
		self.buffer['action'][self.step % self.save_per_step] = action
		self.buffer['reward'][self.step % self.save_per_step] = reward
		self.buffer['done'][self.step % self.save_per_step] = done
		self.step += 1
		if self.step % self.save_per_step == 0:
			self.save()
	def __del__(self):
		self.f.close()
		
class TrajectoryLoader:
	def __init__(self, fname):
		self.f = h5py.File(fname, "a")
		self.dset = {
			'current': self.f['current'][:],
			'next': self.f['next'][:],
			'action': self.f['action'][:],
			'reward': self.f['reward'][:],
			'done': self.f['done'][:]
		}
		self.N = self.dset['current'].shape[0]
	def __del__(self):
		self.f.close()
	def sample(self, n):
		#idx = np.random.random_integers(0, N-1, size=n)
		idx = random.sample(xrange(self.N), n)
		#@ersin - test icin degisik epsilon araliklari secildi
		#idx = random.sample(xrange(150000), n)
		idx = np.array(idx, dtype='int')
		# alttaki satir da test icin
		#idx += 50000
		idx.sort()
		#print(idx)
		return {
			'current': self.dset['current'][idx, :],
			'next': self.dset['next'][idx, :],
			'action': self.dset['action'][idx, :],
			'reward': self.dset['reward'][idx, :],
			'done': self.dset['done'][idx, :]
		}
	def all(self):
		return {
			'current': self.dset['current'][:, :],
			'next': self.dset['next'][:, :],
			'action': self.dset['action'][:, :],
			'reward': self.dset['reward'][:, :],
			'done': self.dset['done'][:, :]
		}

from utils.memory import BaseSampler

class TrajectoryReplay(BaseSampler, RunnerListener):
	def __init__(self, fname, batch_size = 32, UPDATE_FREQUENCY = 1, REPLAY_START_SIZE = -1):
		self.batch_size = batch_size
		self.loader = TrajectoryLoader(fname)
		self.total_step_count = 0
		self.UPDATE_FREQUENCY = UPDATE_FREQUENCY
		self.REPLAY_START_SIZE = REPLAY_START_SIZE
	def get_sample(self):
		samples = self.loader.sample(self.batch_size)
		samples = [
			{
				'current_state': samples['current'][I,:],
				'next_state': samples['next'][I,:],
				'action': samples['action'][I][0],
				'reward': samples['reward'][I][0],
				'done': samples['done'][I][0]
			}
			for I in range(samples['current'].shape[0])
		]
		#print(samples)
		return samples
	def get_all(self):
		samples = self.loader.all()
		samples = [
			{
				'current_state': samples['current'][I,:],
				'next_state': samples['next'][I,:],
				'action': samples['action'][I][0],
				'reward': samples['reward'][I][0],
				'done': samples['done'][I][0]
			}
			for I in range(samples['current'].shape[0])
		]
		return samples
	def has_sample(self):
		return self.total_step_count % self.UPDATE_FREQUENCY == 0 and self.total_step_count>self.REPLAY_START_SIZE	
	def nstep(self):
		return False
	def on_step(self, ob, action, next_ob, reward, done):
		self.total_step_count += 1
