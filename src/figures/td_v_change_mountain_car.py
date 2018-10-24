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

import keras
import keras.backend as K

colors = ['blue', 'red', 'green', 'brown', 'gray', 'yellow', 'cyan', 'purple']
counter = 0

init_nn_library(True, "0")

modelOps = DqnOps(2)
modelOps.INPUT_SIZE = (4,)

env = get_env("MountainCar-v0", False, [])
envOps = EnvOps(env.observation_space.shape, env.action_space.n, 0)

env_model = EnvModelMountainCarManual(envOps)
#env_model.model.load_weights('../test_cartpole_model3/reward-25/weights_100000.h5')
#v_model = CartPoleVNetwork(envOps)
v_model = MountainCarVNetwork(envOps)


v_in = v_model.model.input
v_out = v_model.model.output

td = None
td = TDNetwork(env_model.model, v_model, envOps, include_best_action=True)
#v_in = td.td_model.input
#v_out = td.td_model.get_layer('est_v').output

grads = K.gradients(v_out, v_in)[0]
#grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
#print(grads[0])
get_grads = K.function([v_in], [v_out,grads])

best_act = td.td_model.get_layer('best_act').output
get_best_act = K.function([td.td_model.input], [best_act])


basedir = 'td_offline_v_change_mountaincar9-12'
if os.path.exists(basedir):
	shutil.rmtree(basedir)
os.makedirs(basedir)

def plot_log(fname):
	global counter
	N = 20000
	#tl = TrajectoryLoader(fname)
	#s = tl.sample(N)
	#print(s['current'])
	smin = [-1.2, -0.07]
	smax = [0.5, 0.07]
	M = 30
	L = 100
	samples = np.random.uniform(smin, smax, size=(N,len(smin)))
	X, Y = np.meshgrid(np.linspace(smin[0], smax[0], M), np.linspace(smin[1], smax[1], M))
	X1, Y1 = np.meshgrid(np.linspace(smin[0], smax[0], L), np.linspace(smin[1], smax[1], L))
	z1 = np.stack([X1, Y1], axis=2)
	z1 = np.reshape(z1, (L*L, 2))
	samples = z1

	s = {}
	#print(X, Y)
	s['current'] = samples
	X = np.reshape(X, (M*M, ))
	Y = np.reshape(Y, (M*M, ))
	z = np.stack([X, Y], axis=1)
	#z = np.reshape(z, (M*M, 2))
	print('z',z.shape)
	print(z)
	print('current',s['current'].shape)
	res = env_model.predict_next(samples)
	a = np.random.randint(0, 3, (samples.shape[0],))
	s['next'] = res[0]
	s['next'][a==1] = res[1][a==1]
	s['next'][a==2] = res[2][a==2]
	s['done'] = res[4]
	s['done'][a==1] = res[5][a==1]
	s['done'][a==2] = res[6][a==2]
	
	for step,ID in zip(range(100, 10001, 100), range(0, 10000)):
		#v_model.model.load_weights('../test_cartpole_td/test39/weights_{}.h5'.format(step))
		v_model.model.load_weights('algo_convergence_mountain_car9/td-12/train-0/weights_{}.h5'.format(step))
		if td is not None:
			td.v_model_eval.model.load_weights('algo_convergence_mountain_car9/td-12/train-0/weights_{}.h5'.format(step))
		s['qvalue'] = np.squeeze(v_model.v_value(s['current'])) #[:,1]#
		ns = env_model.predict_next(z)
		best_acts = get_best_act([z])[0]
		best_acts = best_acts.astype('int32')
		#print(ns[0:2], best_acts)
		s['act0'] = ns[0]
		s['act1'] = ns[1]
		s['act2'] = ns[2]
		s['best_act'] = np.array(ns[0])
		#print(ns[1].shape,s['best_act'].shape, s['best_act'][best_acts == 1,:].shape, s['best_act'][best_acts == 2,:].shape)
		#print(best_acts == 1)
		s['best_act'][best_acts == 1] = ns[1][best_acts == 1]
		s['best_act'][best_acts == 2] = ns[2][best_acts == 2]
		#print(best_next)

		s['td_error'] = np.squeeze(td.test(s['current'])[0])
		print(np.mean(s['td_error'][s['td_error']>0]))
		print(np.mean(s['td_error'][s['td_error']<0]))
		print(np.max(s['td_error']))
		print(np.min(s['td_error']))
		_,g1 = get_grads([z])
		#print(g3)
		g2 = g1
		grad = g2
		grad[:,0] = grad[:,0] * (smax[0] - smin[0])
		grad[:,1] = grad[:,1] * (smax[1] - smin[1])
		g3 = np.linalg.norm(grad, axis=1)
		clipVal = 1
		divider = np.reshape(g3, (g3.shape[0],1))[g3>clipVal]
		grad[g3>clipVal,:] = grad[g3>clipVal,:]/divider
		print(divider.shape, g3.shape, g2.shape,g2[g3>clipVal,:].shape)
		
		print('STEP', step)
		for I in range(s['next'].shape[1]):
			for J in range(s['next'].shape[1]):
				if I<J: #and J<K:# and I==0 and J==1: # and I==0 and J==2
					fig = plt.figure(figsize=(16, 8), facecolor='w', edgecolor='k') #num=I*16+J*4, dpi=80, 
					plt.subplot(121)
					i1 = s['done'].flatten() == False
					i2 = s['done'].flatten() == True
					plt.scatter(s['current'][:,I], s['current'][:,J], s=30, c=s['qvalue'][:], alpha=1, edgecolors='none', cmap=plt.get_cmap('viridis'))
					plt.scatter(s['next'][i2,I], s['next'][i2,J], s=1, c=colors[counter*2+1], alpha=1)
					
					Q = plt.quiver(z[:,0], z[:,1], grad[:,0], grad[:,1], units='xy', angles='uv', scale_units='xy', scale=100)
					#Q = plt.quiver(z[:,0], z[:,1], s['act2'][:,0]-z[:,0], s['act2'][:,1]-z[:,1], units='xy', angles='xy', scale_units='xy', scale=1, color='green')
					#Q = plt.quiver(z[:,0], z[:,1], s['act1'][:,0]-z[:,0], s['act1'][:,1]-z[:,1], units='xy', angles='xy', scale_units='xy', scale=1, color='gray')
					#Q = plt.quiver(z[:,0], z[:,1], s['act0'][:,0]-z[:,0], s['act0'][:,1]-z[:,1], units='xy', angles='xy', scale_units='xy', scale=1, color='cyan')
					Q = plt.quiver(z[:,0], z[:,1], s['best_act'][:,0]-z[:,0], s['best_act'][:,1]-z[:,1], units='xy', angles='xy', scale_units='xy', scale=1, color='red')
					
					#qk = plt.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E', coordinates='figure')					
					
					plt.title('{} x {}'.format(I, J))
					ax = plt.gca()
					#ax.set_facecolor((0.0, 0.0, 0.0))
					plt.subplot(122)
					plt.scatter(s['current'][:,I], s['current'][:,J], s=30, c=s['td_error'][:], alpha=1, edgecolors='none', cmap=plt.get_cmap('RdYlBu'), vmin=-0.05, vmax=0.05)
					
					plot_dir = '{}/{}-{}'.format(basedir, I, J)
					if not os.path.exists(plot_dir):
						os.makedirs(plot_dir)
					plt.savefig('{}/{}.png'.format(plot_dir, ID))
					plt.close()
	counter += 1
				
plot_log('../test_tensorboard/traj-9_mix.h5')
