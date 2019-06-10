from runner.runner import RunnerListener

#		from gym.envs.classic_control import rendering
#		self.viewer = rendering.SimpleImageViewer()
#		self.viewer.imshow(ob)

class EnvViewer(RunnerListener):
	def __init__(self, env, render_step = 4, mode='human'):
		self.env = env
		self.render_step = render_step
		self.total_step = 0
		self.mode = mode
	def render(self):
		self.env.render(self.mode)
	def on_step(self, ob, action, next_ob, reward, done):
		self.total_step	+= 1
		if self.total_step % self.render_step == 0:
			self.render()

import numpy as np
from PIL import Image

def save_image(outmap, batch_size, fname):
    output_image = np.copy(outmap)
    output_image[output_image>1] = 1
    output_image[output_image<0] = 0
    output_image = output_image * 255.0
    #outmap[outmap>1] = 1
    #outmap[outmap<0] = 0
    #if len(outmap.shape) == 4:
    #	output_image = outmap[:,:,:,:] * 255.0
    #else:
    #	output_image = outmap[:,:,:] * 255.0
    output_image = output_image.astype(np.uint8)
    if output_image.shape[3] == 1:
        output_image = np.squeeze(output_image,axis=3)
    for J in range(batch_size):
        if len(output_image.shape) == 4:
            output_im = Image.fromarray(output_image[J,:,:,:])
        else:
            output_im = Image.fromarray(output_image[J,:,:])
        output_im.save(fname + f'_{J}.png')
    