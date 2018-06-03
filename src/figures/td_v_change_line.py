import argparse

parser = argparse.ArgumentParser(description='td v change line')
parser.add_argument('--basedir', type=str, default=None, help='base directory to save')
parser.add_argument('--prefix', type=str, default=None, help='prefix of weight files ')
parser.add_argument('--idx', type=int, nargs='*', default=None, help='id from')

args = parser.parse_args()

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

env = get_env("Line", False, [])
envOps = EnvOps(env.observation_space.shape, env.action_space.n, 0)

#env_model = EnvModelLine(envOps)
#env_model.model.load_weights('../test_line_env/lineenv-15/weights_20000.h5')

env_model = EnvModelLineManual(envOps)

v_model = LineVNetwork(envOps)
td_model = TDNetwork(env_model.model, v_model, envOps)

#basedir = 'td_v_change_line_40'
#basedir = 'td_offline_v_change_line_12'
basedir = args.basedir

if os.path.exists(basedir):
	shutil.rmtree(basedir)
os.makedirs(basedir)

def plot_log(fname):
	global counter
	#tl = TrajectoryLoader(fname)
	##N = 300000
	##s = tl.sample(N)
	#s = tl.all()
	#a1 = s['action'].flatten() == 0
	#a2 = s['action'].flatten() == 1
	#a3 = s['current'].flatten() > 0
	#
	##print(s['current'][a2,0], s['reward'][a2][:].flatten())
	#a = np.column_stack((s['current'][:,0], s['reward'][:][:].flatten()))
	#print(s['current'][a3].shape)
	##print(a[14000:14100])
    #
	#x = np.linspace(-1, 1, 1000)
	#res = env_model.predict_next(x)
	##plt.scatter(x, res[:][2][:,0], c=colors[0], s=1)
	#plt.scatter(x, res[:][4][:,0], c=colors[1], s=1)
	##plt.scatter(s['current'][a1,0],s['reward'][a1][:], s=5, alpha=0.4, edgecolors='none', c=colors[2])
	#plt.scatter(s['current'][a2,0],s['done'][a2][:], s=5, alpha=0.4, edgecolors='none', c=colors[3])
	#ac = plt.gca()
	#ac.grid()
	#plt.savefig('test1.png')
	#fig = plt.figure()
	#plt.scatter(x, res[:][0][:,0], c=colors[0], s=1)
	#plt.scatter(x, res[:][1][:,0], c=colors[1], s=1)
	#plt.scatter(s['current'][a1,0],s['next'][a1][:], s=5, alpha=0.4, edgecolors='none', c=colors[2])
	#plt.scatter(s['current'][a2,0],s['next'][a2][:], s=5, alpha=0.4, edgecolors='none', c=colors[3])
	#ac = plt.gca()
	#ac.grid()
	#plt.savefig('test2.png')
	#return
	s = {}
	s['current'] = np.expand_dims(np.linspace(-1, 1, 1000), axis=1)
	print(s['current'].shape)
	for step,ID in zip(range(args.idx[0], args.idx[2], args.idx[1]), range(0, 10000)):
		#v_model.model.load_weights('../test_line_td_realtime/test-40/weights_{}.h5'.format(step))
		#v_model.model.load_weights('../test_line_td/linetd-12/weights_{}.h5'.format(step))
		v_model.model.load_weights((args.prefix + '{}.h5').format(step))
		td_model.v_model_eval.model.set_weights(v_model.model.get_weights())
		s['tderror'] = np.squeeze(td_model.test(s['current']))
		s['qvalue'] = np.squeeze(v_model.v_value(s['current'])) #[:,1]#
		#s['qvalue'] = q_value.max(axis=1) #[:,1]##
		#s['value1'] = q_value[:,0]
		#s['value2'] = q_value[:,1]
		print('STEP', step)
		for I in range(s['current'].shape[1]):
			fig = plt.figure(figsize=(8, 8), facecolor='w', edgecolor='k') #num=I*16+J*4, 
			#i1 = s['done'].flatten() == False
			#i2 = s['done'].flatten() == True
			#plt.scatter(s['current'][:,I],s['value1'][:], s=5, alpha=0.4, edgecolors='none', c='yellow')
			#plt.scatter(s['current'][:,I],s['value2'][:], s=5, alpha=0.4, edgecolors='none', c='green')
			plt.scatter(s['current'][:,I],s['qvalue'][:], s=3, alpha=1, label='state value') #, edgecolors='none'
			plt.scatter(s['current'][:,I],s['tderror'][:], s=3, alpha=1, c=colors[1], label='TD-error') #, edgecolors='none'
			#plt.scatter(s['next'][i2,I], np.zeros((np.sum(i2),)), s=1, c=colors[counter*2+1], alpha=1)

			plt.xlim((-1.1, 1.1))
			plt.ylim((-0.1, 1.1))
			plt.title('Dim. {}'.format(I))
			plt.suptitle('Step {}'.format(step))
			
			ax = plt.gca()
			ax.set_facecolor((0.0, 0.0, 0.0))
			ax.legend()
			plot_dir = '{}'.format(basedir)
			if not os.path.exists(plot_dir):
				os.makedirs(plot_dir)
			plt.savefig('{}/{}.png'.format(plot_dir, ID))
			plt.close()
	counter += 1
				
plot_log('../test_line_dqn/traj-3b_mix.h5')

