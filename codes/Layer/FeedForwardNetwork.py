import torch
import torch.nn as nn
import torch.nn.functional as F


__author__ = "Serena Khoo"


class FeedForwardNetwork(nn.Module):

	"""
		Based on the paper, each layer has 2 subayers:
			A multi-headed attention mechanism & 
			a position-wise fully connected feed-forward network 

		Each layer employs a residual connection, y = f(x) + id(x) = f(x) + x, followed by layer normalization
		This python file would define the position-wise fully connected feed-forward network:
			
			A two layer feed-forward module 
			FFN(x) = max(0, x* w_1 + b_1) * w_2 + b_2
		
	"""
	
	def __init__(self, config):
		
		super().__init__()

		self.config = config
		self.d_model = self.config.d_model
		self.d_feed_forward = self.config.d_feed_forward

		self.w_1 = nn.Linear(self.d_model, self.d_feed_forward)
		self.w_2 = nn.Linear(self.d_feed_forward, self.d_model)

		self.non_linearity = nn.ReLU()
		self.layer_norm = nn.LayerNorm(normalized_shape = self.d_model) # Takes in the normalized shape as input --> normalized shape = d_model (output layer)
		self.dropout = nn.Dropout(p = self.config.dropout_rate, inplace = True)

		# <----------- Initialization of weights ----------->
		nn.init.xavier_normal_(self.w_1.weight)
		nn.init.xavier_normal_(self.w_2.weight)

	def forward(self, x):

		"""
		
			FFN(x) = max(0, x* w_1 + b_1) * w_2 + b_2
			a residual connection, y = f(x) + id(x) = f(x) + x

		"""

		# <----------- Layer 1 ----------->
		output_layer_1 = self.w_1(x)
		output_layer_1 = self.non_linearity(output_layer_1)
		self.dropout(output_layer_1)

		# <----------- Layer 2 ----------->
		output_layer_2 = self.w_2(output_layer_1)

		del output_layer_1
		torch.cuda.empty_cache()

		# <----------- Applying the dropout then layer norm ----------->
		self.dropout(output_layer_2)
		final_output = self.layer_norm(output_layer_2 + x)

		del output_layer_2
		del x
		torch.cuda.empty_cache()

		return final_output