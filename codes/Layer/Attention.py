import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__author__ = "Serena Khoo"

class Attention(nn.Module):

	"""
		This class defines the dot-product attention 
	"""

	def __init__(self, config, d_model, n_head):

		super(Attention, self).__init__()
		self.config = config
		self.d_model = d_model
		self.n_head = n_head

	def forward(self, query, key, val, key_structure = None, val_structure = None, attention_mask = None):

		# <--------- Check that the last dim of the key is the same as d_model / n_head -------------->
		d_k = query.shape[-1]
		assert d_k == self.d_model / self.n_head
		
		# <--------- Calculating the attention, the attention is per head -------------->
		attention_values = torch.matmul(query, key.transpose(len(key.shape) -2, len(key.shape) -1)) #QK'

		del key
		torch.cuda.empty_cache()

		# <-------- Adding the edge information -------------->
		if key_structure is not None:

			edge_score = torch.matmul(query.unsqueeze(3), key_structure.transpose(4,3)).squeeze(3)
			attention_values = attention_values + edge_score

			del edge_score
			del key_structure
			torch.cuda.empty_cache()

		del query
		torch.cuda.empty_cache()

		# <-------- Scaling of the attention values -------------->
		scaling_factor = np.power(d_k, 0.5) # d_k**0.5

		# <-------- Apply attention masking if attention_mask is not None -------------->
		if attention_mask is not None:

			attention_mask = attention_mask.unsqueeze(1).unsqueeze(len(attention_mask.shape))
			attention_mask = (1.0 - attention_mask) * -100000.0

			attention_values = attention_values + attention_mask

			del attention_mask
			torch.cuda.empty_cache()

		# <-------- Getting the softmax of the attention values -------------->
		attention_values = attention_values / scaling_factor
		attention_values = F.softmax(attention_values, dim = -1)

		# <-------- Getting the final output -------------->
		final_output = torch.matmul(attention_values, val)

		del val
		torch.cuda.empty_cache()

		# <-------- Getting the edge scores to be added to val -------------->
		if val_structure is not None:

			edge_val_score = torch.matmul(attention_values.unsqueeze(3), val_structure).squeeze(3)

			final_output = final_output + edge_val_score

			del val_structure
			del edge_val_score
			torch.cuda.empty_cache()

		return final_output, attention_values

