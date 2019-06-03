import tensorflow as tf
import tensorflow.keras.backend as K

class SummaryWriter(object):
	def __init__(self, sw, scalar_names, model = None, histogram_freq = 0, write_grads = False):
		self.sw = sw
		self.scalar_names = scalar_names
		self.scalar_vars = [tf.Variable(0.0) for _ in range(len(scalar_names))]
		self.summary_op = tf.summary.merge([tf.summary.scalar(name, self.scalar_vars[i]) for i,name in enumerate(scalar_names)])
		self.model = model
		self.histogram_freq = histogram_freq
		self.write_grads = write_grads
		# from keras source code (https://github.com/keras-team/keras/blob/master/keras/callbacks.py)
		self.histogramOps = []
		self.histogram_op = None
		#if self.histogram_freq > 0 and self.model is not None:
		#	for layer in self.model.layers:
        #        for weight in layer.weights:
        #            mapped_weight_name = weight.name.replace(':', '_')
        #            .append(tf.summary.histogram(mapped_weight_name, weight))
        #            if self.write_grads:
        #                grads = model.optimizer.get_gradients(model.total_loss, weight)
        #                def is_indexed_slices(grad):
        #                    return type(grad).__name__ == 'IndexedSlices'
        #                grads = [
        #                    grad.values if is_indexed_slices(grad) else grad
        #                    for grad in grads]
        #                self.histogramOps.append(tf.summary.histogram('{}_grad'.format(mapped_weight_name), grads))
		#	self.histogram_op = tf.summary.merge(self.histogramOps)
		
	def add(self, scalar_vals, step):
		sess = K.get_session()
		sum_str = sess.run(self.summary_op, feed_dict = dict((a, scalar_vals[i]) for i, a in enumerate(self.scalar_vars)))
		if self.sw is not None:
			self.sw.add_summary(sum_str, step)
			self.sw.flush()
