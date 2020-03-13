import torch 
import torch.nn as nn
import torch.nn.functional as F
from Layer import Layer

__author__ = "Serena Khoo"

class Transformer(nn.Module):

	@staticmethod
	def init_weights(layer):
		if type(layer) == nn.Linear:
			nn.init.xavier_normal_(layer.weight)

	def __init__(self, config, n_layers, d_model, n_heads):

		super(Transformer, self).__init__()

		# <----------- Config ----------->
		self.config = config

		# <----------- Model dimensions ----------->
		self.n_layers = n_layers
		self.d_model = d_model
		self.n_heads = n_heads

		# <----------- Stack of Attention layers ----------->
		self.input_stack = nn.ModuleList([Layer.Layer(config, d_model, n_heads) for _ in range(n_layers)])

	def forward(self, query, key, val, key_structure = None, val_structure = None, attention_mask = None):

		"""

		This function takes in a sequence and apply MHA to it

		"""

		# Merge with the query at each layer
		self_atten_output = query
		del query
		torch.cuda.empty_cache()

		# Storing the attention weights at each layer 
		self_atten_weights_dict = {}
		i = 1

		# Passing through the MHA layers
		for layer in self.input_stack:

			self_atten_output, self_atten_weights = layer(query = self_atten_output, 
														  key = key,
														  val = val,
														  key_structure = key_structure,
														  val_structure = val_structure,
														  attention_mask = attention_mask)

			self_atten_weights_dict[i] = self_atten_weights
			i += 1
			del self_atten_weights
			torch.cuda.empty_cache()

		return self_atten_output, self_atten_weights_dict

	def __repr__(self):
		return str(vars(self))
