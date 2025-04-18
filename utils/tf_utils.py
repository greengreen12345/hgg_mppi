import numpy as np
# import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

# def get_vars(scope_name):
# 	vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)
# 	assert len(vars) > 0
# 	return vars


class Normalizer_torch:
	def __init__(self, shape, device, eps_std=1e-2, norm_clip=5):
		self.shape = shape
		self.device = device
		self.eps_std = torch.tensor(eps_std).to(self.device)
		self.norm_clip = norm_clip

		self.sum = torch.zeros(self.shape, dtype=torch.float32, requires_grad=False).to(self.device)
		self.sum_sqr = torch.zeros(self.shape, dtype=torch.float32, requires_grad=False).to(self.device)
		self.cnt = torch.zeros(1, dtype=torch.float32, requires_grad=False).to(self.device)
		self.mean = torch.zeros(self.shape, dtype=torch.float32, requires_grad=False).to(self.device)
		self.std = torch.zeros(self.shape, dtype=torch.float32, requires_grad=False).to(self.device)

	def get_mean(self):
		return self.mean.numpy()

	def get_std(self):
		return self.std.numpy()

	def normalize(self, inputs_ph):
		return torch.clamp((inputs_ph - self.mean) / self.std, -self.norm_clip, self.norm_clip)

	def normalizer_update(self, batch):
		self.update(np.concatenate([batch['obs'], batch['obs_next']], axis=0))

	def update(self, inputs):
		inputs = torch.Tensor(inputs).to(self.device)
		add_sum = torch.sum(inputs, dim=0)
		add_sum_sqr = torch.sum(torch.square(inputs), dim=0)
		add_cnt = torch.tensor([inputs.shape[0]], dtype=torch.float32).to(self.device)

		self.sum += add_sum
		self.sum_sqr += add_sum_sqr
		self.cnt += add_cnt

		self.mean = self.sum / self.cnt
		self.std = torch.maximum(self.eps_std, torch.sqrt(self.sum_sqr / self.cnt - torch.square(self.sum / self.cnt)))

class Normalizer:
	def __init__(self, shape, sess, eps_std=1e-2, norm_clip=5):
		self.shape = shape
		self.sess = sess
		self.eps_std = eps_std
		self.norm_clip = norm_clip

		with tf.variable_scope('normalizer_variables', initializer=tf.zeros_initializer()):
			self.sum = tf.get_variable(name='sum', shape=self.shape, dtype=np.float32, trainable=False)
			self.sum_sqr = tf.get_variable(name='sum_sqr', shape=self.shape, dtype=np.float32, trainable=False)
			self.cnt = tf.get_variable(name='cnt', shape=[1], dtype=np.float32, trainable=False)
			self.mean = tf.get_variable(name='mean', shape=self.shape, dtype=np.float32, trainable=False)
			self.std = tf.get_variable(name='std', shape=self.shape, dtype=np.float32, trainable=False)

		self.add_sum = tf.placeholder(tf.float32, self.shape)
		self.add_sum_sqr = tf.placeholder(tf.float32, self.shape)
		self.add_cnt = tf.placeholder(tf.float32, [1])

		self.update_array_op = tf.group(
			self.sum.assign_add(self.add_sum),
			self.sum_sqr.assign_add(self.add_sum_sqr),
			self.cnt.assign_add(self.add_cnt)
		)
		self.update_scalar_op = tf.group(
			self.mean.assign(self.sum/self.cnt),
			self.std.assign(tf.maximum(self.eps_std, tf.sqrt(self.sum_sqr/self.cnt-tf.square(self.sum/self.cnt))))
		)

	def get_mean(self): return self.sess.run(self.mean)
	def get_std(self): return self.sess.run(self.std)

	def normalize(self, inputs_ph):
		return tf.clip_by_value((inputs_ph-self.mean)/self.std, -self.norm_clip, self.norm_clip)

	def update(self, inputs):
		feed_dict = {
			self.add_sum: np.sum(inputs, axis=0),
			self.add_sum_sqr: np.sum(np.square(inputs), axis=0),
			self.add_cnt: [inputs.shape[0]]
		}
		self.sess.run(self.update_array_op, feed_dict)
		self.sess.run(self.update_scalar_op)
