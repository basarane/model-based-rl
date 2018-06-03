import matplotlib 
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, '../')

import utils.trajectory_utils
reload(utils.trajectory_utils)
from utils.trajectory_utils import TrajectoryLoader, TrajectoryReplay
from nets.net import *

import shutil
import os

colors = ['blue', 'red', 'green', 'brown', 'gray', 'yellow', 'cyan', 'purple']
counter = 0

init_nn_library(True, "1")

modelOps = DqnOps(2)
modelOps.INPUT_SIZE = (1,)
q_model = LineModel(modelOps)

basedir = 'dqn_v_change_line'
if os.path.exists(basedir):
	shutil.rmtree(basedir)
os.makedirs(basedir)

def plot_log(fname):
	global counter
	N = 1000
	tl = TrajectoryLoader(fname)
	#tr = TrajectoryReplay(fname)
	#s = tl.sample(N)
	#s = tl.all()
	s = {
		'current': np.linspace(-1, 1, N).reshape((N,1))
	}
	#print(s)
	#return
	#print(s['current'])
	print(s['current'].shape)
	for step,ID in zip(range(100, 50001, 100), range(0, 100000)):
		#q_model.model.load_weights('../test_line_dqn/linedqn-2/weights_{}.h5'.format(step))
		q_model.model.load_weights('algo_convergence_line/dqn/train-0/weights_{}.h5'.format(step))
		
		q_value = q_model.q_value(s['current'])
		s['qvalue'] = q_value.max(axis=1) #[:,1]##
		s['value1'] = q_value[:,0]
		s['value2'] = q_value[:,1]
		print('STEP', step)
		for I in range(s['current'].shape[1]):
			fig = plt.figure(figsize=(8, 8), facecolor='w', edgecolor='k') #num=I*16+J*4,  #figsize=(16, 16), dpi=80, 
			#i1 = s['done'].flatten() == False
			#i2 = s['done'].flatten() == True
			plt.scatter(s['current'][:,I],s['value1'][:], s=5, alpha=0.7, edgecolors='none', c='yellow', label='value(right)')
			plt.scatter(s['current'][:,I],s['value2'][:], s=5, alpha=0.7, edgecolors='none', c='red', label='value(left)')
			#plt.scatter(s['current'][:,I],s['qvalue'][:], s=1, alpha=0.4, edgecolors='none')
			#plt.scatter(s['next'][i2,I], np.zeros((np.sum(i2),)), s=1, c=colors[counter*2+1], alpha=1)
			plt.xlim((-1.1, 1.1))
			plt.ylim((-0.1, 1.1))
			plt.title('Dim. {}'.format(I))
			plt.suptitle('Step {}'.format(step))
			ax = plt.gca()
			ax.set_facecolor((0.0, 0.0, 0.0))
			plot_dir = '{}'.format(basedir)
			if not os.path.exists(plot_dir):
				os.makedirs(plot_dir)
			plt.savefig('{}/{}.png'.format(plot_dir, ID))
			plt.close()
	counter += 1
				
plot_log('../test_line_dqn/traj-1_mix.h5')
