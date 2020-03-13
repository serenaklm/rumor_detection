import torch 
import torch.nn as nn
import torch.nn.functional as F
import os 
import numpy as np

class PositionEncoder(nn.Module):

	"""
	Encodes the information into vectors

	There are 2 pieces of information that goes into the encoded information: 
	1. Word Embedding 
	2. Position Embedding 

	This set of codes would encode the position information
	
	"""

	@staticmethod
	def pos_emb(pos, dim, d_model):

		return pos / np.power(10000, (2 * (dim//2) / d_model))

	@staticmethod
	def cal_pos_emb(pos, d_emb_dim, d_model):

		
		return [PositionEncoder.pos_emb(pos, dim, d_model) for dim in range(d_emb_dim)] # Calculates for each dim 

	@staticmethod
	def get_position_embedding(max_index, d_model, d_emb_dim, requires_grad = False):

		# position_embedding --> [[sin(0),cos(1),sin(2),cos(3)...], [sin(0),cos(1),sin(2),cos(3)...] ...] 
		
		position_embedding = np.array([PositionEncoder.cal_pos_emb(pos, d_emb_dim, d_model) for pos in range(max_index + 1)]) # Getting for each token

		position_embedding[:, 0::2] = np.sin(position_embedding[:, 0::2])  # dim 2i --> Starts from 0 
		position_embedding[:, 1::2] = np.cos(position_embedding[:, 1::2])  # dim 2i+1 --> starts from 1

		position_embedding = torch.FloatTensor(position_embedding)
		position_embedding.requires_grad = requires_grad

		return position_embedding

	def __init__(self, config, max_index):

		super(PositionEncoder, self).__init__()

		self.config = config
		self.max_index = max_index # Including the index
		self.d_model = self.config.d_model
		self.d_emb_dim = self.config.emb_dim
		self.requires_grad = self.config.train_pos_emb
		self.freeze = True if self.requires_grad == False else False

		# <------------- Defining the position embedding -------------> 

		self.pos_embd_weights = PositionEncoder.get_position_embedding(max_index = self.max_index, 
																	   d_model = self.config.d_model, 
																	   d_emb_dim = self.d_emb_dim,
																	   requires_grad = self.requires_grad)

		self.position_encoding = nn.Embedding.from_pretrained(self.pos_embd_weights, freeze = self.freeze) 

	def forward(self, src_seq):

		"""
		Ref:
		https://pytorch.org/docs/stable/nn.html
		
		Does encoding for the input:
		1. position encoding (The position encoding are based on the time stamp)

		<--------- POS Embedding --------->
		Input:
			src_seq :

		Output:
			encoded_pos_features : 

		"""

		position_index = src_seq.long()
		encoded_pos_features = self.position_encoding(position_index)

		return encoded_pos_features