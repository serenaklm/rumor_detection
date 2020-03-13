import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from Encoder import PositionEncoder
from Layer import Attention

__author__ = "Serena Khoo"


class MultiHeadAttention(nn.Module):

	"""
		Based on the paper, each layer has 2 subayers:
			A multi-headed attention mechanism & 
			a position-wise fully connected feed-forward network 

		Each layer employs a residual connection, y = f(x) + id(x) = f(x) + x, followed by layer normalization
		This python file would define the Multi Attention network
		
	"""

	@staticmethod
	def init_weights(layer):
		if type(layer) == nn.Linear:
			nn.init.xavier_normal_(layer.weight)

	def __init__(self, config, d_model, n_head, attention_mask = None):

		super(MultiHeadAttention, self).__init__()

		# <----------- Config ----------->
		self.config = config

		# <----------- Model dimensions ----------->
		self.d_model = d_model
		self.n_head = n_head

		assert self.d_model % self.n_head == 0, print("Word dim cannot be split into {} heads equally".format(self.n_head))

		self.d_k = self.d_model // self.n_head
		self.d_v = self.d_k

		# <----------- Projection layers ----------->
		self.proj_layer_query = nn.ModuleList([nn.Linear(self.config.d_model, self.d_v) for _ in range(self.config.n_head)]) 
		self.proj_layer_key = nn.ModuleList([nn.Linear(self.config.d_model, self.d_v) for _ in range(self.config.n_head)]) 
		self.proj_layer_val = nn.ModuleList([nn.Linear(self.config.d_model, self.d_v) for _ in range(self.config.n_head)])

		# <----------- Attention Layer ----------->
		self.attention = Attention.Attention(self.config, self.d_model, self.n_head)

		# <----------- Layer Norm and FC Layer ----------->
		self.layer_norm = nn.LayerNorm(self.d_model)
		self.fc = nn.Linear(self.d_model, self.d_model)

		# <----------- Drop out ----------->
		self.dropout = nn.Dropout(p = self.config.dropout_rate, inplace = True)

		# <----------- Initialization ----------->
		nn.init.xavier_normal_(self.fc.weight)
		self.proj_layer_query.apply(MultiHeadAttention.init_weights)
		self.proj_layer_key.apply(MultiHeadAttention.init_weights)
		self.proj_layer_val.apply(MultiHeadAttention.init_weights)


	def forward(self, query, key, val, key_structure = None, val_structure = None, attention_mask = None):

		"""

		This function defines the multi head attention network 

		"""

		# <--------- Setting the residual -------------->
		residual = query

		# <--------- Getting the projections for each head -------------->

		if self.config.gpu == True:

			query_head = Variable(torch.zeros((self.n_head, *query.shape[:-1], self.d_k), device = torch.device("cuda")))
			key_head = Variable(torch.zeros((self.n_head, *query.shape[:-1], self.d_k), device = torch.device("cuda")))
			val_head = Variable(torch.zeros((self.n_head, *query.shape[:-1], self.d_k), device = torch.device("cuda")))

		else:
			query_head = Variable(torch.zeros((self.n_head, *query.shape[:-1], self.d_k)))
			key_head = Variable(torch.zeros((self.n_head, *query.shape[:-1], self.d_k)))
			val_head = Variable(torch.zeros((self.n_head, *query.shape[:-1], self.d_k)))


		for i in range(self.n_head):
			query_head[i] =  self.proj_layer_query[i](query).unsqueeze(0)
			key_head[i] =  self.proj_layer_key[i](key).unsqueeze(0)
			val_head[i] =  self.proj_layer_val[i](val).unsqueeze(0)

		# <--------- Clear the memory -------------->
		del query
		del key 
		del val
		torch.cuda.empty_cache()

		# <--------- Move the batch to be the first dimension --------->
		query_head = query_head.permute(1,0,*(np.arange(2,len(query_head.shape)))).contiguous()
		key_head = key_head.permute(1,0,*(np.arange(2,len(query_head.shape)))).contiguous()
		val_head = val_head.permute(1,0,*(np.arange(2,len(query_head.shape)))).contiguous()

		# <--------- Getting the attention values -------------->
		if key_structure is not None and val_structure is not None:

			self_atten_features, atten_values = self.attention(query_head, key_head, val_head, key_structure = key_structure, val_structure = val_structure, attention_mask = attention_mask)

		else:

			self_atten_features, atten_values = self.attention(query_head, key_head, val_head, attention_mask = attention_mask)

		# <--------- Clear the memory -------------->
		del query_head
		del key_head 
		del val_head
		torch.cuda.empty_cache()

		# <--------- Projecting back to full d_model -------------->
		num_dim = len(self_atten_features.shape)

		self_atten_features = self_atten_features.permute(0, *(np.arange(2, num_dim -1)), 1, num_dim -1).contiguous()
		self_atten_features = self_atten_features.view(*(self_atten_features.shape[:-2]), -1)
		self_atten_features = self.fc(self_atten_features)

		# <--------- Applying the dropout then layer norm -------------->
		self.dropout(self_atten_features)

		self_atten_features = self.layer_norm(self_atten_features + residual)

		return self_atten_features, atten_values