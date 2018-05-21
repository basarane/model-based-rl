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

import shutil
import os

colors = ['blue', 'red', 'green', 'brown', 'gray', 'yellow', 'cyan', 'purple']
counter = 0

init_nn_library(True, "1")

modelOps = DqnOps(2)
modelOps.INPUT_SIZE = (4,)
q_model = CartPoleModel(modelOps)

basedir = 'dqn_v_change_cartpole'
if os.path.exists(basedir):
	shutil.rmtree(basedir)
os.makedirs(basedir)

def plot_log(fname):
	global counter
	N = 300000
	tl = TrajectoryLoader(fname)
	s = tl.sample(N)
	#print(s['current'])
	print(s['current'].shape)
	for step,ID in zip(range(100, 30000, 100), range(0, 10000)):
		q_model.model.load_weights('../test_cartpole2/dqn-15/weights_{}.h5'.format(step))
		s['qvalue'] = q_model.q_value(s['current']).max(axis=1)#[:,1]#
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
	counter += 1
				
plot_log('../test_tensorboard/traj-9_mix.h5')
