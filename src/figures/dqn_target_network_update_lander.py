import sys
sys.path.insert(0, '../')

from algo.dqn import run_dqn

arguments = {
	'game': 'LunarLander-v2',
	'mode': 'train',
	'model': 'LunarLanderModel',
	'learning_rate': 0.0001,
	'target_network_update': 5000,
	'egreedy_final': [0.01],
	'egreedy_decay': [1],
	'egreedy_props': [1],
	'egreedy_final_step': [250000],
	'env_transforms': [],
	'replay_start_size': 5000,
	'replay_buffer_size': 100000, 
	'batch_size': 64, 
	'update_frequency': 1, 
	'max_step': 1000000,
	'logdir': 'dqn_target_network_update_lander/test-1',
	'dont_init_tf': True,
	#defaults
	'double_dqn': False, 
	'dueling_dqn': False, 
	'atari': False,
	'output_dir': None,
	'enable_render': False,
	'load_weightfile': None,
	'test_epsilon': 0.01
}

import numpy as np

import tensorflow as tf
import keras.backend as K
from nets.net import init_nn_library
import pickle
import shutil
import os

baseDir = 'dqn_target_network_update_lander'


def run_dqn_thread(baseDir, runNo, thread_id, args):
	init_nn_library(True, '0')
	runner, agent = run_dqn(**args)
	runner.run()
	with open(baseDir + '/' + str(args['target_network_update']) + '-' + str(runNo) + '.pkl', 'w') as f:
		pickle.dump(agent.stats, f)

from threading import Thread

target_network_updates = [100, 300, 1000, 3000, 10000, 30000]

import time

from multiprocessing import Process

if __name__ == '__main__':
	if os.path.exists(baseDir):
		shutil.rmtree(baseDir)
	os.makedirs(baseDir)
	for runNo in range(4):
		procs = []
		for I in range(len(target_network_updates)):
			args = arguments.copy()
			args['logdir'] = baseDir + '/rbs-' + str(target_network_updates[I]) + '-' + str(runNo) 
			args['target_network_update'] = target_network_updates[I]
			proc = Process(target=run_dqn_thread, args=(baseDir, runNo, I, args))
			#time.sleep(1)
			proc.start()
			procs.append(proc)
		for proc in procs:
			proc.join()
			