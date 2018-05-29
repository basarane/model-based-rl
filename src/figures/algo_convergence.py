import sys
sys.path.insert(0, '../')

from algo.dqn import run_dqn, run_dqn_test
from algo.td import run_td, run_td_test
from algo.td_realtime import run_td_realtime, run_td_realtime_test

from nets.net import init_nn_library
import pickle
import shutil
import os

def run_dqn_thread(baseDir, runNo, args, max_episode):
	init_nn_library(True, '1')
	runner, _ = run_dqn(**args)
	runner.run()
	run_test('dqn', args, baseDir, runNo, max_episode)

def run_td_thread(baseDir, runNo, args, max_episode):
	init_nn_library(True, '1')
	stats = run_td(**args)
	run_test('td', args, baseDir, runNo, max_episode)

def run_td_realtime_thread(baseDir, runNo, args, max_episode):
	init_nn_library(True, '1')
	runner, _ = run_td_realtime(**args)
	runner.run()
	run_test('td_realtime', args, baseDir, runNo, max_episode)
	
def run_test(algo, args, baseDir, runNo, max_episode):
	test_args = args.copy()
	test_args['mode'] = 'test'
	test_args['load_weightfile'] = [test_args['logdir'] + "/weights_", test_args['save_interval'], test_args['save_interval'], test_args['max_step']]
	test_args['max_episode'] = max_episode
	test_args['max_step'] = 1000000000
	test_args['logdir'] = baseDir + '/' + algo + '/test-' + str(runNo) 
	stats = globals()['run_' + algo + '_test'](**test_args)
	with open(test_args['logdir'] + '/final_stats.pkl', 'w') as f:
		pickle.dump(stats, f)

import time

from multiprocessing import Process

def algo_convergence(arguments, baseDir, algos, run_count = 4, max_episode = 5, delete_if_exists = True):
	if delete_if_exists and os.path.exists(baseDir):
		shutil.rmtree(baseDir)
	if not os.path.exists(baseDir):
		os.makedirs(baseDir)
	for runNo in range(run_count):
		procs = []
		for algo in algos:
			args = arguments.copy()
			args['logdir'] = baseDir + '/' + algo + '/train-' + str(runNo) 
			proc = Process(target=globals()['run_' + algo + '_thread'], args=(baseDir, runNo, args, max_episode))
			proc.start()
			procs.append(proc)
	for proc in procs:
		proc.join()
			