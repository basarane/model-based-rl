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
	'save_interval': 10000,
	'save_freq': 10000,
	'vmodel': 'CartPoleVNetwork',
	'env_model': 'EnvModelMountainCarManual',
	'env_weightfile': None,
	'smin': [-1.2, -0.07],
	'smax': [0.5, 0.07],
	'td_exponent': 2,
	'target_network_update': 30,
	'sample_count': 20000,
	'load_trajectory': None,
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
basedir = 'algo_convergence_mountain_car'
#tddir = basedir + '/td'
#if os.path.exists(tddir):
#	shutil.rmtree(tddir)
#os.makedirs(tddir)

algo_convergence(arguments, basedir, ['td'], run_count = 4, max_episode = 5, delete_if_exists = False, test_only=True)
 #'dqn', 'td', 'td_realtime'
 