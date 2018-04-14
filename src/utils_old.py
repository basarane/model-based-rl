import keras.backend as K
import numpy as np
from PIL import Image, ImageDraw

def get_activations(model, model_inputs, print_shape_only=False, layer_name=None):
	print('----- activations -----')
	activations = []
	inp = model.input

	model_multi_inputs_cond = True
	if not isinstance(inp, list):
		# only one input! let's wrap it in a list.
		inp = [inp]
		model_multi_inputs_cond = False

	#from pprint import pprint
	#pprint(vars(model.layers[3]))

	for layer in model.layers:
		print(layer.name, len(layer.outbound_nodes), len(layer.inbound_nodes))
		for I in range(len(layer.inbound_nodes)):
			o1 = layer.get_output_at(I)
			print(o1.name, o1.shape)
			
	outputs = [[layer.get_output_at(I) for I in range(len(layer.inbound_nodes))] for layer in model.layers if (layer.name == layer_name or layer_name is None)]
	outputs = [item for sublist in outputs for item in sublist]
	#outputs.extend([])

	funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

	if model_multi_inputs_cond:
		list_inputs = []
		list_inputs.extend(model_inputs)
		list_inputs.append(0.)
	else:
		list_inputs = [model_inputs, 0.]

	print("model_multi_inputs_cond", model_multi_inputs_cond, len(list_inputs))
	# Learning phase. 0 = Test mode (no dropout or batch normalization)
	# layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]
	layer_outputs = [func(list_inputs)[0] for func in funcs]
	for layer_activations in layer_outputs:
		activations.append(layer_activations)
		if print_shape_only:
			print(layer_activations.shape)
		else:
			print(layer_activations)
	return activations

def toRGBImage(x):
	im = Image.fromarray(x)
	im = im.convert('RGB') 
	return np.array(im, dtype='uint8')

def	prediction_to_image(prediction, meanImage):
	predOutput = np.array(prediction)*255.0
	predOutput = predOutput + meanImage
	predOutput[predOutput<0] = 0
	predOutput[predOutput>255] = 255
	predOutput = np.array(predOutput, dtype="uint8")
	predImage = np.squeeze(predOutput)
	return predImage
	
def draw_reward(predImage, reward):
	im = Image.fromarray(predImage)
	draw = ImageDraw.Draw(im)
	w = 100
	x = 57
	draw.rectangle([x,196,x+int(w*reward),208], "#fff", None)
	draw.rectangle([x,196,x+w,208], None, "#f00")
	predImage = np.array(im)
	return predImage

def get_obs_input(lastFramesOrig, meanImage):
	netin = np.array(lastFramesOrig, dtype='f')/255.0
	netin = np.squeeze(netin)
	netin = np.transpose(netin, (0,3,1,2))
	netin = np.reshape(netin, (12, 210,160))
	netin = netin - np.tile(np.transpose(meanImage/255.0, (2,0,1)), (4,1,1))
	netin = np.reshape(netin, (1, 12, 210,160))
	return netin
	