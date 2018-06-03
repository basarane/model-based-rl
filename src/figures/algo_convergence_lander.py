#python -B oo_dqn.py LunarLander-v2 --model LunarLanderModel --learning-rate 0.00025 --target-network-update 5000 --egreedy-final 0.01 --egreedy-final-step 250000 --egreedy-decay 1 --replay-start-size 5000 --replay-buffer-size 50000 --batch-size 64 --update-frequency 1 --max-step 1000000 --logdir test_lunarlander/dqn-9

#--nstep 5 --egreedy-props 0.4 0.3 0.3 --thread-count 8

arguments = {
	'game': 'LunarLander-v2',
	'mode': 'train',
	'model': 'LunarLanderModel',
	'learning_rate': 0.00025,
	'egreedy_final': [0.01],
	'egreedy_decay': [1],
	'egreedy_props': [1],
	'egreedy_final_step': [250000],
	'env_transforms': ['Penalizer'], #
	'replay_start_size': 5000,
	'replay_buffer_size': 50000, 
	'batch_size': 64, 
	'update_frequency': 1, 
	'max_step': 1000000,
	'max_episode': None,
	'logdir': 'dqn_td_convergence_cartpole/test-1',
	'dont_init_tf': True,
	'save_interval': 2000,
	'save_freq': 2000,
	#'vmodel': 'CartPoleVNetwork',
	#'env_model': 'EnvModelCartPoleManual',
	#'env_weightfile': None,
	#'smin': [-2.4, -1, -0.20943, -1],
	#'smax': [2.4, 1, 0.20943, 1],
	#'td_exponent': 1,
	#'sample_count': 20000,
	'target_network_update': 5000,
	'load_trajectory': None,
	#defaults
	'double_dqn': False, 
	'dueling_dqn': False, 
	'atari': False,
	'output_dir': None,
	'enable_render': False,
	'render_step': 1,
	'load_weightfile': None,
	'test_epsilon': 0.00
}

from algo_convergence import algo_convergence
import os
import shutil
basedir = 'algo_convergence_lander'
tddir = basedir + '/td'
if os.path.exists(tddir):
	shutil.rmtree(tddir)
os.makedirs(tddir)

algo_convergence(arguments, basedir, [
	{
		'algo': 'dqn',
		'dir': 'dqn',
		'args': {
		}
	}, 
	{
		'algo': 'dqn',
		'dir': 'ddqn',
		'args': {
			'double_dqn': True
		}
	}, 
	{
		'algo': 'dqn',
		'dir': 'dueling-dqn',
		'args': {
			'dueling_dqn': True
		}
	}, 
	{
		'algo': 'a3c',
		'dir': 'a3c',
		'args': {
			'nstep': 5,
			'egreedy_props': [5, 5, 5], 
			'egreedy_final': [0.01, 0.1, 0.5], 
			'egreedy_decay': [1, 1, 1],
			'egreedy_final_step': [30000, 30000, 30000],
			'thread_count': 16,
			'target_network_update': 100
		}
	}], run_count = 1, max_episode = 1, delete_if_exists = True, test_only=False)
 #'dqn', 'td', 'td_realtime'
 