arguments = {
	'game': 'Line',
	'mode': 'train',
	'model': 'LineModel',
	'learning_rate': 0.0003,
	'target_network_update': 30,
	'egreedy_final': [0.01],
	'egreedy_decay': [1],
	'egreedy_props': [1],
	'egreedy_final_step': [10000],
	'env_transforms': [],
	'replay_start_size': 64,
	'replay_buffer_size': 10000, 
	'batch_size': 64, 
	'update_frequency': 1, 
	'max_step': 50000,
	'max_episode': None,
	'logdir': 'dqn_td_convergence_line/test-1',
	'dont_init_tf': True,
	'save_interval': 100,
	'save_freq': 100,
	'vmodel': 'LineVNetwork',
	'env_model': 'EnvModelLineManual',
	'env_weightfile': None,
	'smin': [-1],
	'smax': [1],
	'sample_count': 5000,
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

algo_convergence(arguments, 'algo_convergence_line', ['dqn', 'td', 'td_realtime'], run_count = 4, max_episode = 5)
