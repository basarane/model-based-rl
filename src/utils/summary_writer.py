import tensorflow as tf
import keras.backend as K

class SummaryWriter(object):
	def __init__(self, sw, scalar_names):
		self.sw = sw
		self.scalar_names = scalar_names
		self.scalar_vars = [tf.Variable(0.0) for _ in range(len(scalar_names))]
		self.summary_op = tf.summary.merge([tf.summary.scalar(name, self.scalar_vars[i]) for i,name in enumerate(scalar_names)])
		
	def add(self, scalar_vals, step):
		sess = K.get_session()
		sum_str = sess.run(self.summary_op, feed_dict = dict((a, scalar_vals[i]) for i, a in enumerate(self.scalar_vars)))
		if self.sw is not None:
			self.sw.add_summary(sum_str, step)
		
