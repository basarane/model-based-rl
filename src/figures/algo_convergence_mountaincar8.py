arguments = {
	'game': 'MountainCar-v0',
	'mode': 'train',
	'model': 'CartPoleModel',
	'learning_rate': 0.0003,
	'egreedy_final': [0.01],
	'egreedy_decay': [0.999],
	'egreedy_props': [1],
	'egreedy_final_step': [10000],
	'env_transforms': [], #
	'replay_start_size': 1000,
	'replay_buffer_size': 10000, 
	'batch_size': 64, 
	'update_frequency': 1, 
	'max_step': 50000,
	'max_episode': None,
	'logdir': 'dqn_td_convergence_cartpole/test-1',
	'dont_init_tf': True,
	'save_interval': 100,
	'save_freq': 100,
	'vmodel': 'MountainCarVNetwork', #CartPoleVNetwork
	'env_model': 'EnvModelMountainCarManual',
	'env_weightfile': None,
	'smin': [-1.2, -0.07],
	'smax': [0.5, 0.07],
	'td_exponent': 2,
	'target_network_update': 30,
	'sample_count': 20000,
	'load_trajectory': None,
	'gpu': '0',
	#defaults
	'double_dqn': False, 
	'dueling_dqn': False, 
	'atari': False,
	'output_dir': None,
	'enable_render': False,
	'load_weightfile': None,
	'test_epsilon': 0.00
}

	
from algo_convergence import algo_convergence
import os
import shutil

basedir = 'algo_convergence_mountain_car8'

algo_convergence(arguments, basedir, [
	{
		'algo': 'td',
		'dir': 'td-9',
		'args': {
			'learning_rate': 0.0003,
			'td_exponent': 2,
			'target_network_update': 25,
			'gpu': '0',
		}
	},
	{
		'algo': 'td',
		'dir': 'td-10',
		'args': {
			'learning_rate': 0.0003,
			'td_exponent': 2,
			'target_network_update': 75,
			'gpu': '0',
		}
	},
	{
		'algo': 'td',
		'dir': 'td-11',
		'args': {
			'learning_rate': 0.0003,
			'td_exponent': 2,
			'target_network_update': 200,
			'gpu': '0',
		}
	},
	#{
	#	'algo': 'td',
	#	'dir': 'td-1',
	#	'args': {
	#		'learning_rate': 0.0003,
	#		'td_exponent': 2,
	#		'target_network_update': "('linear', 1, 500, 50000)",
	#		'gpu': '0',
	#	}
	#},
	#{
	#	'algo': 'td',
	#	'dir': 'td-2',
	#	'args': {
	#		'learning_rate': 0.0003,
	#		'td_exponent': 1,
	#		'target_network_update': "('linear', 1, 500, 50000)",
	#		'gpu': '0',
	#	}
	#},
	#{
	#	'algo': 'td',
	#	'dir': 'td-3',
	#	'args': {
	#		'learning_rate': 0.0003,
	#		'td_exponent': 0,
	#		'target_network_update': "('linear', 1, 500, 50000)",
	#		'gpu': '0',
	#	}
	#},
	#{
	#	'algo': 'td',
	#	'dir': 'td-4',
	#	'args': {
	#		'learning_rate': 0.0003,
	#		'td_exponent': "('linear', 0, 2, 30000)",
	#		'target_network_update': "('linear', 1, 500, 50000)",
	#		'gpu': '0',
	#	}
	#},
	#{
	#	'algo': 'td',
	#	'dir': 'td-5',
	#	'args': {
	#		'learning_rate': 0.0003,
	#		'td_exponent': "('linear', 0, 2, 10000)",
	#		'target_network_update': "('linear', 1, 500, 50000)",
	#		'gpu': '0',
	#	}
	#},
	#{
	#	'algo': 'td',
	#	'dir': 'td-6',
	#	'args': {
	#		'learning_rate': 0.0003,
	#		'td_exponent': "('linear', 0, 2, 10000)",
	#		'target_network_update': "('linear', 1, 100, 50000)",
	#		'gpu': '1',
	#	}
	#},
	#{
	#	'algo': 'td',
	#	'dir': 'td-7',
	#	'args': {
	#		'learning_rate': 0.0003,
	#		'td_exponent': "('linear', 0, 4, 10000)",
	#		'target_network_update': "('linear', 1, 500, 50000)",
	#		'gpu': '1',
	#	}
	#},
	#{
	#	'algo': 'td',
	#	'dir': 'td-8',
	#	'args': {
	#		'learning_rate': 0.0003,
	#		'td_exponent': "('linear', 0, 1.5, 10000)",
	#		'target_network_update': "('linear', 1, 50, 50000)",
	#		'gpu': '1',
	#	}
	#},

], run_count = 1, max_episode = 1, delete_if_exists = False, test_only=False)
 