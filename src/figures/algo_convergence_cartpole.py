#python -B oo_dqn.py CartPole-v1 --model CartPoleModel --learning-rate 0.0003 --target-network-update 30 --egreedy-final 0.01 --egreedy-decay 0.999 --env-transforms Penalizer --replay-start-size 1000 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 30000 --save-interval 100 --logdir test_cartpole2/dqn-15

arguments = {
	'game': 'CartPole-v1',
	'mode': 'train',
	'model': 'CartPoleModel',
	'learning_rate': 0.0003,
	'egreedy_final': [0.01],
	'egreedy_decay': [0.999],
	'egreedy_props': [1],
	'egreedy_final_step': [10000],
	'env_transforms': ['Penalizer'], #
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
	'vmodel': 'CartPoleVNetwork',
	'env_model': 'EnvModelCartPoleManual',
	'env_weightfile': None,
	'smin': [-2.4, -1, -0.20943, -1],
	'smax': [2.4, 1, 0.20943, 1],
	'td_exponent': 1,
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

# old tests (in reverse order)
#	'smin': [-2, -3, -0.3, -3],  td-01
#	'smax': [2, 3, 0.3, 3],
#
#	'smin': [-2, -3, -0.3, -0.2],
#	'smax': [2, 3, 0.3, 0.2],
#
#
#	'smin': [-2, -3, -2, -0.2],  td-02
#	'smax': [2, 3, 2, 0.2],
#
#
#	'smin': [-2, -0.2, -2, -0.2], td-03
#	'smax': [2, 0.2, 2, 0.2],
#
#	'smin': [-2, -0.2, -2, -0.2], td-04
#	'smax': [2, 0.2, 2, 0.2],
#	'td_exponent': 1,
#
#	'smin': [-2.4, -1, -0.20943, -1], td-05
#	'smax': [2.4, 1, 0.20943, 1],
#	'td_exponent': 1,
#
#	'smin': [-2.4, -0.2, -0.20943, -0.2], td-06
#	'smax': [2.4, 0.2, 0.20943, 0.2],
#	'td_exponent': 1,
#	'target_network_update': 30,
#
#	'smin': [-2.4, -0.2, -0.20943, -0.2], td-07
#	'smax': [2.4, 0.2, 0.20943, 0.2],
#	'td_exponent': 1,
#	'target_network_update': 250,
#
#	'smin': [-2.4, -3, -0.20943, -3], td-08  (td-09: same parameters with huber_loss_mse and RMSprop)
#	'smax': [2.4, 3, 0.20943, 3],
#	'td_exponent': 1,
#	'target_network_update': 250,
#
#	'smin': [-2.4, -1, -0.20943, -1], td-10 (td-05 ile ayni parametreler, huber_loss_mse and RMSprop)
#	'smax': [2.4, 1, 0.20943, 1],
#	'td_exponent': 1,
#	'target_network_update': 30,
#
#	'smin': [-2.4, -1, -0.20943, -1], td-11 (td-05 ile ayni parametreler, mse and RMSprop)
#	'smax': [2.4, 1, 0.20943, 1],
#	'td_exponent': 1,
#	'target_network_update': 30,

#	'smin': [-3, -0.2, -2, -0.2],
#	'smax': [3, 0.2, 2, 0.2],
#
#	'smin': [-3, -0.2, -2, -0.5],
#	'smax': [3, 0.2, 2, 0.5],
#
#	'smin': [-3, -0.5, -2, -0.5],
#	'smax': [3, 0.5, 2, 0.5],

	
from algo_convergence import algo_convergence
import os
import shutil
basedir = 'algo_convergence_cartpole_01'
tddir = basedir + '/td'
if os.path.exists(tddir):
	shutil.rmtree(tddir)
os.makedirs(tddir)

algo_convergence(arguments, basedir, ['td'], run_count = 4, max_episode = 5, delete_if_exists = False)
 #'dqn', 'td', 'td_realtime'
 