import matplotlib
#%matplotlib inline
matplotlib.use('agg')
import matplotlib.pyplot as plt
import glob
import os
import itertools
from operator import itemgetter
import pickle
import numpy as np

import sys
sys.path.insert(0, '../')
from utils.smooth import smooth

def plot_perf(basedir, plot_title, x_axis, N, n):
	files = glob.glob(basedir + '/*.pkl')
	runs = []
	for full_name in files:
		fname = os.path.splitext(os.path.basename(full_name))[0]
		#print(fname)
		p = float(fname.rsplit('-',1)[0])
		with open(full_name, 'r') as f:
			rewards = pickle.load(f)['reward']
		rewards = np.array(rewards)
		idx = rewards[:,0] >= n
		aver = np.sum(rewards[idx,1]) / (N - n)
		runs.append({
			'p': p,
			'fname': full_name,
			'rewards': rewards,
			'aver_reward': aver
		})
	runs_sorted = sorted(runs, key=itemgetter('p'))
	stats = []
	for k, g in itertools.groupby(runs_sorted, key=lambda x: x['p']):
		#print(k)
		r = np.array([a['aver_reward'] for a in list(g)])
		stats.append([k, r.mean(), r.std()])

	stats = np.array(stats)
	#print(stats)
	plt.figure()
	plt.errorbar(stats[:,0], stats[:,1], yerr=stats[:,2],fmt='o-')
	plt.title(plot_title)
	plt.xlabel(x_axis)
	plt.ylabel('average reward per step')
	ax = plt.gca()
	ax.set_xscale('log') 
	plt.savefig(basedir + '.png')
	plt.savefig(basedir + '.pdf')
	
	
	
plot_perf('dqn_replay_memory_size_cartpole', 'Cart pole', 'replay memory size', 100000, 90000)
plot_perf('dqn_learning_rate_cartpole', 'Cart pole', 'learning rate', 100000, 50000)
plot_perf('dqn_target_network_update_cartpole', 'Cart pole', 'target network update', 100000, 50000)
plot_perf('dqn_replay_memory_size_lander', 'Lunar Lander', 'replay memory size', 1000000, 900000)
plot_perf('dqn_learning_rate_lander', 'Lunar Lander', 'learning rate', 1000000, 500000)
plot_perf('dqn_target_network_update_lander', 'Lunar Lander', 'target network update', 1000000, 500000)
plot_perf('td_realtime_target_network_update_line', 'line', 'target network update', 1000000, 0)

def plot_compare(basedir, algos, fname, title):
	plt.figure()
	for I,algo in enumerate(algos):
		print(I, algo)
		files = glob.glob(basedir + "/" + algo + "/test-*/final_stats.pkl")
		aver_rewards = None
		for full_name in files:
			print(full_name)
			with open(full_name, 'r') as f:
				rewards = pickle.load(f)['reward']
			if aver_rewards is None:
				aver_rewards = np.array(rewards)
				print(aver_rewards.shape)
			else:
				aver_rewards[:,1] += np.array(rewards)[:,1]
		if aver_rewards is not None:
			aver_rewards[:, 1] = aver_rewards[:, 1] / len(files)
			algo_title = algo
			if algo_title == 'dqn':
				algo_title = 'DQN'
			if algo_title == 'a3c':
				algo_title = 'Async n-step Q (A3C paper)'
			if algo_title == 'ddqn':
				algo_title = 'Double-DQN'
			if algo_title == 'dueling-dqn':
				algo_title = 'Dueling-DQN'
			if algo_title == 'td':
				algo_title = 'Offline-TD'
			if algo_title == 'td_realtime':
				algo_title = 'Online-TD'
			plt.plot(aver_rewards[:, 0],smooth(aver_rewards[:, 1], 9), label=algo_title)
	#plt.errorbar(stats[:,0], stats[:,1], yerr=stats[:,2],fmt='o-')
	#plt.title(plot_title)
	#plt.xlabel(x_axis)
	#plt.ylabel('average reward per step')
	ax = plt.gca()
	plt.legend()
	plt.title(title)
	plt.xlabel('step')
	plt.ylabel('total episode reward')
	#ax.set_xscale('log') 
	plt.savefig(basedir + '_' + fname + ".png")
	plt.savefig(basedir + '_' + fname + ".pdf")
	
plot_compare('algo_convergence_line', ['dqn', 'td_realtime', 'td'], 'test', 'Line') #
plot_compare('algo_convergence_cartpole', ['dqn', 'td_realtime', 'td'], 'test', 'Cartpole') #'td', 'td_realtime', 
plot_compare('algo_convergence_cartpole_01', ['td-01', 'td-02', 'td-03', 'td-04', 'td-05', 'td-06', 'td-07', 'td-08', 'td-09', 'td-10', 'td'], 'test-td', 'Cartpole') 
plot_compare('algo_convergence_cartpole_01', ['dqn', 'td_realtime', 'td'], 'test', 'Cartpole') #'td', 'td_realtime', 

plot_compare('algo_convergence_lander', ['dqn', 'ddqn', 'dueling-dqn', 'a3c'], 'test', 'Lunar Lander') #