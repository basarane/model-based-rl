import sys
sys.path.insert(0, '../')

from algo.dqn import run_dqn, run_dqn_test
from algo.td import run_td, run_td_test
from algo.td_realtime import run_td_realtime, run_td_realtime_test
from algo.a3c import run_a3c

from nets.net import init_nn_library
import pickle
import shutil
import os

def run_dqn_thread(baseDir, runNo, args, max_episode, test_only):
	init_nn_library(True, args['gpu'])
	if not test_only:
		runner, _ = run_dqn(**args)
		runner.run()
	run_test('dqn', args, baseDir, runNo, max_episode)

def run_a3c_thread(baseDir, runNo, args, max_episode, test_only):
	init_nn_library(True, args['gpu'])
	if not test_only:
		stats = run_a3c(**args)
	#with open(baseDir + '/test-' + str(runNo)  + '/final_stats.pkl', 'w') as f:
	#	pickle.dump(stats, f)	
	run_test('dqn', args, baseDir, runNo, max_episode)

def run_td_thread(baseDir, runNo, args, max_episode, test_only):
	print('*************GPU: *************** ' + args['gpu'])
	init_nn_library(True, args['gpu'])
	if not test_only:
		stats = run_td(**args)
	run_test('td', args, baseDir, runNo, max_episode)

def run_td_realtime_thread(baseDir, runNo, args, max_episode, test_only):
	init_nn_library(True, args['gpu'])
	if not test_only:
		runner, _ = run_td_realtime(**args)
		runner.run()
	run_test('td_realtime', args, baseDir, runNo, max_episode)
	
def run_test(algo, args, baseDir, runNo, max_episode):
	test_args = args.copy()
	test_args['mode'] = 'test'
	test_args['load_weightfile'] = [test_args['logdir'] + "/weights_", test_args['save_interval'], test_args['save_interval'], test_args['max_step']]
	test_args['max_episode'] = max_episode
	test_args['max_step'] = 1000000000
	test_args['logdir'] = baseDir + '/test-' + str(runNo) #'/' + algo +
	stats = globals()['run_' + algo + '_test'](**test_args)
	with open(test_args['logdir'] + '/final_stats.pkl', 'w') as f:
		pickle.dump(stats, f)

import time

from multiprocessing import Process

def algo_convergence(arguments, baseDir, algos, run_count = 4, max_episode = 5, delete_if_exists = True, test_only=False):
	if delete_if_exists and os.path.exists(baseDir) and not test_only:
		shutil.rmtree(baseDir)
	if not os.path.exists(baseDir):
		os.makedirs(baseDir)
	for runNo in range(run_count):
		procs = []
		for algo in algos:
			args = arguments.copy()
			
			if type(algo) is dict:
				algoname = algo['algo']
				dirname = algo['dir']
				args.update(algo['args'])
			else:
				algoname = algo
				dirname = algo
			print(baseDir, dirname, runNo, args) #, algo['args']
			args['logdir'] = baseDir + '/' + dirname + '/train-' + str(runNo) 
			proc = Process(target=globals()['run_' + algoname + '_thread'], args=(baseDir + '/' + dirname, runNo, args, max_episode, test_only))
			proc.start()
			procs.append(proc)
	for proc in procs:
		proc.join()
			