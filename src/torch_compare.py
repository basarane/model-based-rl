import numpy as np

def torch_compare(torch_compare_path, ACTION_COUNT, model, model_eval, MINIBATCH_SIZE,AGENT_HISTORY_LENGTH, INPUT_SIZE, DISCOUNT_FACTOR)
	for J in range(1004, 1024, 4):
		print('TRAIN', J)
		
		loadBase = args.torch_compare_path+str(J)+'/'

		ts, ts2, ta, tr, tterm, results, dw, delta, deltas, g, g2, targets, tmp = load_step(loadBase, ACTION_COUNT, model, model_eval)	#, prediction, prediction_layer
		
		current_state = np.zeros((MINIBATCH_SIZE,AGENT_HISTORY_LENGTH)+INPUT_SIZE, dtype='f')
		next_state = np.zeros((MINIBATCH_SIZE,AGENT_HISTORY_LENGTH)+INPUT_SIZE, dtype='f')
			
		for I in xrange(MINIBATCH_SIZE):
			current_state[I] = ts[I]
			next_state[I] = ts2[I]

		target = model.predict(current_state)
		targetx = target.copy()
		next_value = model_eval.predict(next_state)
		if args.double_dqn:
			next_best_res = model.predict(next_state)
			best_acts = np.argmax(next_best_res, axis=1)
		else:
			best_acts = np.argmax(next_value, axis=1)
		#print('Next_Value: ', next_value)
		for I in xrange(MINIBATCH_SIZE):
			action = ta[I]
			reward = tr[I]
			if tterm[I]==1:
				target[I,action] = reward
			else:
				target[I,action] = reward + DISCOUNT_FACTOR * next_value[I,best_acts[I]]   #after double DQN

		#print('Targets calculated: ', target- targetx)
		#print('Target adifference: ', target- targetx - targets)
		res = model.train_on_batch(current_state, target)
		di = 9
		print('dw', dw[di])
		print('delta', deltas[di])
		print('g', g[di])
		print('g2', g2[di])
		print('tmp', tmp[di])
		#print('targets', targets)
		#print(K.get_value(my_optimizer.grads[0]))
		print("second train")
		#res = model.train_on_batch(current_state, target)
