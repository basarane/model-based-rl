import numpy as np
from PIL import Image

def load_torch_tensor(filename, idx = 0, lineno = 0):
	if idx>0:
		lineno = 2+21*idx
	with open(filename) as fp:
		for i, line in enumerate(fp):
			if i == lineno:
				a = np.fromstring(line, dtype='float', sep= ' ')
	return a

def load_torch_network_parameter(filename, model, idx = 0, lineno = 0):
	params = load_torch_tensor(filename, idx, lineno)
	#print('PARAMCOUNT', params.shape)
	w = model.get_weights()
	idx = 0
	for a in w:
		c = a.size
		#print(a.shape)
		if len(a.shape)==4:
			a[:] = np.transpose(params[idx:(idx+c)].reshape(a.shape[::-1]), [2,3,1,0])
			#print(a[2,1,1,0])
		elif len(a.shape)==2:
			a[:] = np.transpose(params[idx:(idx+c)].reshape(a.shape[::-1]), [1,0])
			#print(a[1,0])
		else:
			a[:] = params[idx:(idx+c)]
			#print(a[1])
		idx += c
	return w
def load_torch_model(filename, model):
	w = load_torch_network_parameter(filename, model, 1)
	model.set_weights(w)

def load_transitions(filename):
	a = load_torch_tensor(filename, 1)
	term = load_torch_tensor(filename, 2)
	r = load_torch_tensor(filename, 3)[:32]
	return a, r, term

def load_step(loadBase, ACTION_COUNT, model, model_eval):
	load_torch_model(loadBase + 'tokeras.network.params.t7', model)
	if not model_eval is None:
		load_torch_model(loadBase + 'tokeras.target.params.t7', model_eval)

	global ts, ts2, ta, tr, tterm, results, dw, delta, deltas, g, g2, targets, tmp, prediction, prediction_layer
	ts = []
	for I in range(32):
		im1 = Image.open(loadBase + 'image-s-'+str(I+1)+'-0.png')
		im2 = Image.open(loadBase + 'image-s-'+str(I+1)+'-1.png')
		im3 = Image.open(loadBase + 'image-s-'+str(I+1)+'-2.png')
		im4 = Image.open(loadBase + 'image-s-'+str(I+1)+'-3.png')
		ts.append([np.array(im1), np.array(im2), np.array(im3), np.array(im4)])
	#print(ts)
	ts = np.array(ts, dtype='f')/255.0
	ts2 = []
	for I in range(32):
		im1 = Image.open(loadBase + 'image-s2-'+str(I+1)+'-0.png')
		im2 = Image.open(loadBase + 'image-s2-'+str(I+1)+'-1.png')
		im3 = Image.open(loadBase + 'image-s2-'+str(I+1)+'-2.png')
		im4 = Image.open(loadBase + 'image-s2-'+str(I+1)+'-3.png')
		ts2.append([np.array(im1), np.array(im2), np.array(im3), np.array(im4)])
	#print(ts)
	ts2 = np.array(ts2, dtype='f')/255.0
	ta, tr, tterm = load_transitions(loadBase + 'tokeras.trans.t7')
	ta = ta.astype('int') - 1
	tterm = tterm.astype('int')

	results = load_torch_tensor(loadBase + 'tokeras.results.t7', lineno=17)
	#dw = load_torch_tensor(loadBase + 'tokeras.dw.t7', lineno=17)
	dw = load_torch_network_parameter(loadBase + 'tokeras.dw.t7', model, lineno=17)
	delta = load_torch_tensor(loadBase + 'tokeras.delta.t7', lineno=17)
	deltas = load_torch_network_parameter(loadBase + 'tokeras.deltas.t7', model, lineno=17)
	g = load_torch_network_parameter(loadBase + 'tokeras.g.t7', model, lineno=17)
	g2 = load_torch_network_parameter(loadBase + 'tokeras.g2.t7', model, lineno=17)
	targets = load_torch_tensor(loadBase + 'tokeras.targets.t7', lineno=17)
	tmp = load_torch_network_parameter(loadBase + 'tokeras.tmp.t7', model, lineno=17)

	results = np.resize(results, (32,ACTION_COUNT))
	targets = np.resize(targets, (32,ACTION_COUNT))

	#if not model_eval is None:
	#	prediction = model_eval.predict_on_batch(ts2)
	#	prediction_layer = model_layer.predict_on_batch(ts2)
	#	#print('Prediction difference: ',prediction-results)
	print('Shapes: ', results.shape, len(dw), len(deltas), delta.shape, targets.shape, len(tmp))
	return ts, ts2, ta, tr, tterm, results, dw, delta, deltas, g, g2, targets, tmp #, prediction, prediction_layer