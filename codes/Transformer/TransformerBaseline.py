import torch 
import torch.nn as nn
import torch.nn.functional as F
from Transformer import Transformer

__author__ = "Serena Khoo"

class TransformerBaseline(nn.Module):

	@staticmethod
	def init_weights(layer):
		if type(layer) == nn.Linear:
			nn.init.xavier_normal_(layer.weight)

	def __init__(self, config):

		super(TransformerBaseline, self).__init__()

		self.config = config

		# <----------- Getting a transformer (post level) ----------->
		self.transformer_post = Transformer.Transformer(self.config, self.config.n_mha_layers, self.config.d_model, self.config.n_head)

		# <----------- Embedding the key, val and query ----------->
		self.emb_layer_query = nn.ModuleList([nn.Linear(self.config.emb_dim, self.config.emb_dim) for _ in range(self.config.num_emb_layers)])
		self.emb_layer_val = nn.ModuleList([nn.Linear(self.config.emb_dim, self.config.emb_dim) for _ in range(self.config.num_emb_layers)])
		self.emb_layer_key = nn.ModuleList([nn.Linear(self.config.emb_dim, self.config.emb_dim) for _ in range(self.config.num_emb_layers)])

		# <----------- Layer Normalization ----------->
		self.layer_norm = nn.LayerNorm(normalized_shape = self.config.emb_dim) 
		
		# <----------- Final layer to predict the output class (4 classes) -----------> 
		self.final_layer = nn.Sequential(nn.Linear(self.config.d_model, self.config.num_classes),
										 nn.LogSoftmax(dim = 1))

		# <----------- Initialization of weights ----------->
		self.emb_layer_query.apply(Transformer.init_weights)
		self.emb_layer_val.apply(Transformer.init_weights)
		self.emb_layer_key.apply(Transformer.init_weights)
		self.final_layer.apply(Transformer.init_weights)

	def forward(self, X, time_delay):

		"""

		This function takes in the posts associated with each rumour (Organized in chronological order) and apply multihead attention (MHA) to it 

		1. Max pooling is applied to each post to get the more prevelant feature for each post

		"""

		# <----------- Getting the dimensions ----------->
		batch_size, num_posts, num_words, emb_dim = X.shape

		# <----------- Reshaping X (Combining batch and posts) ----------->
		X = X.view(-1, num_words, emb_dim)
		
		# <----------- Do max pooling for X ----------->
		X = X.permute(0, 2, 1).contiguous()
		X = F.adaptive_max_pool1d(X, 1).squeeze(-1)

		# <----------- Rehshaping X (Shaping them back into batches) ----------->
		X = X.view(batch_size, num_tweets, -1)

		# <----------- Setting the query, key and val ----------->
		query = X 
		key = X
		val = X

		# <----------- Passing the query, key and val with n number of feedforward layers  ----------->
		for i in range(self.config.num_emb_layers):
			query = self.layer_norm(self.emb_layer_query[i](query))
			key = self.layer_norm(self.emb_layer_key[i](key))
			val = self.layer_norm(self.emb_layer_val[i](val))

		# <----------- Adding in time delay information ----------->
		query = query + time_delay
		key = key + time_delay
		val = val + time_delay

		# <----------- Passing through post level transformer (Not keeping the attention values for now) ----------->
		self_atten_output, self_atten_weights_dict = self.transformer_post(query, key, val)

		# Getting the average embedding for the self attended output 
		self_atten_output = self_atten_output.permute(0, 2, 1).contiguous()
		self_atten_output = F.adaptive_max_pool1d(self_atten_output, 1).squeeze(-1)

		# <------- Passing through a feed foward layer then do a softmax to do prediction  ------->
		output = self.final_layer(self_atten_output)

		return output, self_atten_weights_dict

	def __repr__(self):
		return str(vars(self))
