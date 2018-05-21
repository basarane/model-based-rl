import matplotlib 
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, '../')

import utils.trajectory_utils
reload(utils.trajectory_utils)
from utils.trajectory_utils import TrajectoryLoader
from nets.net import *
from envs.env_transform import *
from envs.gym_env import *
from env_model.model import *

import shutil
import os

colors = ['blue', 'red', 'green', 'brown', 'gray', 'yellow', 'cyan', 'purple']
counter = 0

init_nn_library(True, "1")

modelOps = DqnOps(2)
modelOps.INPUT_SIZE = (4,)

env = get_env("CartPole-v1", False, ["Penalizer"])
envOps = EnvOps(env.observation_space.shape, env.action_space.n, 0)

env_model = EnvModelCartpole(envOps)
env_model.model.load_weights('../test_cartpole_model3/reward-25/weights_100000.h5')
v_model = CartPoleVNetwork(envOps)

basedir = 'td_v_change_cartpole'
#if os.path.exists(basedir):
#	shutil.rmtree(basedir)
#os.makedirs(basedir)

def plot_log(fname):
	global counter
	N = 300000
	tl = TrajectoryLoader(fname)
	s = tl.sample(N)
	#print(s['current'])
	print(s['current'].shape)
	for step,ID in zip(range(751, 1000, 1), range(750, 10000)):
		v_model.model.load_weights('../test_cartpole_td/test39/weights_{}.h5'.format(step))
		s['qvalue'] = np.squeeze(v_model.v_value(s['current'])) #[:,1]#
		print('STEP', step)
		for I in range(s['next'].shape[1]):
			for J in range(s['next'].shape[1]):
				if I<J: #and J<K:# and I==0 and J==1: # and I==0 and J==2
					fig = plt.figure(figsize=(16, 16), dpi=80, facecolor='w', edgecolor='k') #num=I*16+J*4, 
					i1 = s['done'].flatten() == False
					i2 = s['done'].flatten() == True
					plt.scatter(s['current'][:,I], s['current'][:,J], s=20, c=s['qvalue'][:], alpha=0.4, edgecolors='none', cmap=plt.get_cmap('viridis'))
					plt.scatter(s['next'][i2,I], s['next'][i2,J], s=1, c=colors[counter*2+1], alpha=1)
					plt.title('{} x {}'.format(I, J))
					ax = plt.gca()
					ax.set_facecolor((0.0, 0.0, 0.0))
					plot_dir = '{}/{}-{}'.format(basedir, I, J)
					if not os.path.exists(plot_dir):
						os.makedirs(plot_dir)
					plt.savefig('{}/{}.png'.format(plot_dir, ID))
					plt.close()
	counter += 1
				
plot_log('../test_tensorboard/traj-9_mix.h5')
