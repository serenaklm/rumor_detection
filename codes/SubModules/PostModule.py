import torch 
import torch.nn as nn
import torch.nn.functional as F
from Transformer import Transformer
from Encoder import LearnedPositionEncoder

__author__ = "Serena Khoo"

class PostModule(nn.Module):

	@staticmethod
	def init_weights(layer):
		if type(layer) == nn.Linear:
			nn.init.xavier_normal_(layer.weight)

	def __init__(self, config):

		super(PostModule, self).__init__()

		# <----------- Config ----------->
		self.config = config
		
		# <----------- Key and val structure encoder ----------->
		if self.config.include_key_structure:
			self.key_structure_encoder = LearnedPositionEncoder.LearnedPositionEncoder(self.config, self.config.n_head)

		if self.config.include_val_structure:
			self.val_structure_encoder = LearnedPositionEncoder.LearnedPositionEncoder(self.config, self.config.n_head)

		# <----------- Getting a transformer for each level (word & post level) ----------->
		self.transformer_post = Transformer.Transformer(self.config, self.config.n_mha_layers, self.config.d_model, self.config.n_head)
		
		# <----------- Embedding the posts through n FC layers (post level) ----------->
		if self.config.ff_post:
			self.emb_layer_post = nn.ModuleList([nn.Linear(self.config.emb_dim, self.config.emb_dim) for _ in range(self.config.num_emb_layers)])

		# <----------- Fine Tunning layer after getting only the first post's embedding  -----------> 
		self.fine_tune_layer = nn.Linear(self.config.emb_dim, self.config.emb_dim)

		# <----------- Final layer to predict the output class (4 classes) (To map emb to classes) -----------> 
		self.final_layer_emb = nn.Sequential(nn.Linear(self.config.emb_dim, self.config.num_classes),
											 nn.LogSoftmax(dim = 1))

		# <----------- To map each post vector to a scalar ----------->
		self.condense_layer_post = nn.Linear(self.config.d_model, 1)

		self.final_layer_posts = nn.Sequential(nn.Linear(self.config.max_tweets, self.config.num_classes),
												nn.LogSoftmax(dim = 1))

		# <----------- Droutput for regularization ----------->
		self.dropout = nn.Dropout(p = self.config.dropout_rate, inplace = True)

		# <----------- Initialization of weights ----------->
		if self.config.ff_post:
			self.emb_layer_post.apply(PostModule.init_weights)

		self.fine_tune_layer.apply(PostModule.init_weights)
		self.final_layer_emb.apply(PostModule.init_weights)
		
		self.condense_layer_post.apply(PostModule.init_weights)

		self.final_layer_posts.apply(PostModule.init_weights)

	def forward(self, X_word, time_delay, batch_size, num_posts, emb_dim, structure = None, attention_mask = None):

		# <----------- Encoding the structure ----------->

		key_structure = None
		val_structure = None

		if self.config.include_key_structure:
			key_structure = self.key_structure_encoder(structure)

		if self.config.include_val_structure:
			val_structure = self.val_structure_encoder(structure)

		# <----------- Passing X_word with n number of FC layers (post level), set on/off in config ----------->
		if self.config.ff_post:
			for i in range(self.config.num_emb_layers):
				X_word = self.emb_layer_post[i](X_word)

		# <----------- Adding in time delay information ----------->
		if self.config.include_time_interval:
			X_word += time_delay

		# <----------- Setting the query, key and val ----------->
		query_post = X_word
		key_post = X_word
		val_post = X_word

		del X_word
		torch.cuda.empty_cache()

		# <----------- Adding in time delay information ----------->
		self.dropout(query_post)
		self.dropout(key_post)
		self.dropout(val_post)

		# <----------- Passing through post level transformer (Not keeping the attention values for now) ----------->
		self_atten_output_post, self_atten_weights_dict_post = self.transformer_post(query_post, key_post, val_post, key_structure = key_structure, val_structure = val_structure, attention_mask = attention_mask)

		# <----------- Baseline: Getting the average embedding for the self attended output ----------->
		if self.config.post_module_version == 0:
			self_atten_output_post = F.adaptive_avg_pool1d(self_atten_output_post.permute(0, 2, 1), 1).squeeze(-1)

			# # <----------- Doing dropout here ----------->
			# self.dropout(self_atten_output_post)

			# <----------- Getting the predictions ----------->
			output = self.final_layer_emb(self_atten_output_post)

			torch.cuda.empty_cache()

		# <----------- Approach 1: Condensing the post features in to a vector with length max_tweets ----------->
		if self.config.post_module_version == 1:
			self_atten_output_post = self.condense_layer_post(self_atten_output_post).squeeze(-1)
			
			# # <----------- Doing dropout here ----------->
			# self.dropout(self_atten_output_post)

			# <----------- Getting the predictions ----------->
			output = self.final_layer_posts(self_atten_output_post)

			torch.cuda.empty_cache()

		# <----------- Approach 2: Just get the first vector (Vector of the source post) ----------->
		if self.config.post_module_version == 2:

			self_atten_output_post = self.fine_tune_layer(self_atten_output_post[:, 0, :])
			
			# # <----------- Doing dropout here ----------->
			# self.dropout(self_atten_output_post)

			# <----------- Getting the predictions ----------->
			output = self.final_layer_emb(self_atten_output_post)

			torch.cuda.empty_cache()

		# <----------- Approach 3: Attention over the vector ----------->
		if self.config.post_module_version == 3:

			attention_mask += -1.0
			attention_mask *= 100000.0

			posts_attention_values = self.condense_layer_post(self_atten_output_post)
			posts_attention_weights = F.softmax(posts_attention_values.permute(0,2,1) + attention_mask.unsqueeze(-2), dim = -1)

			del posts_attention_values
			torch.cuda.empty_cache()

			self_atten_output_post = torch.matmul(posts_attention_weights, self_atten_output_post).squeeze(1)
			
			# # <----------- Doing dropout here ----------->
			# self.dropout(self_atten_output_post)

			# <----------- Getting the predictions ----------->
			output = self.final_layer_emb(self_atten_output_post)

			del attention_mask
			torch.cuda.empty_cache()
		
		return output, posts_attention_weights, self_atten_weights_dict_post
