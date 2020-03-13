import numpy as np

# <------- Self defined classes -------->
from config import config

__author__ = "Serena Khoo"

class Optimizer():

	def __init__(self, config, optimizer):

		super(Optimizer, self).__init__()

		self.config = config
		self.optimizer = optimizer

		self.n_warmup_steps = self.config.n_warmup_steps
		self.n_current_steps = 0
		self.init_lr = self.config.learning_rate #np.power(self.config.d_model, -0.5)

	def get_lr_scale(self):
		return min(np.power(self.n_current_steps, -0.5), self.n_current_steps * np.power(self.n_warmup_steps, -1.5))

	def update_learning_rate(self):

		self.n_current_steps += 1
		lr = self.init_lr * self.get_lr_scale()

		for param_group in self.optimizer.param_groups:
			param_group['lr'] = lr

	def step_and_update_lr(self):
		if self.config.vary_lr:
			self.update_learning_rate()
		self.optimizer.step()

	def zero_grad(self):
		self.optimizer.zero_grad()

	def state_dict(self):
		return self.optimizer.state_dict()