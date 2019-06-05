from PIL import Image
import scipy.misc
import numpy as np

class Preprocessor(object):
	def __init__(self):
		pass
	def preprocess(self, image):
		if isinstance(image, (list,)):
			return [self.preprocess_one(a) for a in image]
		else:
			return self.preprocess_one(image)
	def preprocess_one(self, image):
		raise NotImplementedException()

class GrayPrePro(Preprocessor):
	def __init__(self):
		super(GrayPrePro, self).__init__()
	def preprocess_one(self, image):
		im = Image.fromarray(image)
		im = im.convert('L') 
		return np.array(im, dtype='uint8')

class ResizePrePro(Preprocessor):
	def __init__(self, size):
		super(ResizePrePro, self).__init__()
		self.size = size
	def preprocess_one(self, image):
		return scipy.misc.imresize(image, self.size, interp='bilinear')

class RewardClipper(Preprocessor):
	def __init__(self, min_reward=-1000000, max_reward = 1000000):
		super(RewardClipper, self).__init__()
		self.min_reward = min_reward
		self.max_reward = max_reward
	def preprocess_one(self, val):
		return max(min(val, self.max_reward), self.min_reward)
			
class PreProPipeline(Preprocessor):
	def __init__(self, procs = None):
		self.procs = [] if procs is None else procs
	def add(self, proc):
		self.procs.append(proc)
	def preprocess(self, image):
		for proc in self.procs:
			image = proc.preprocess(image)
		return image

def remove_bg(ob, bg):
    same = (ob[:,:,0] == bg[:,:,0]) & (ob[:,:,1] == bg[:,:,1]) & (ob[:,:,2] == bg[:,:,2])
    ob = np.concatenate((ob, 255*np.ones((210, 160, 1))), axis=2)
    ob[same] = 0
    ob[same,3] = 0
    ob = ob/255.0
    return ob
