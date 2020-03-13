import torch
import torch.nn as nn
from torchtext import vocab
from torchtext.data import NestedField, Field, Pipeline, TabularDataset, BucketIterator
import torch.nn.functional as F

import numpy as np
import os
import re
from tqdm import tqdm

import spacy
from spacy.lang.zh import Chinese
nlp = spacy.load("en", disable = ["parser", "tagger", "ner"])
nlp_chinese = Chinese()

__author__ = "Serena Khoo"

class DataLoader():

	"""
	This is the dataloader class that takes in a path and return a generator that could be iterated through

	init:
		path: path of the data to read in (assumes CSV format)
		config: a Config object that contains the parameters to be used
		shuffle: whether to shuffle the data or not (true by default)

	"""

	def __init__(self, config, split, type_ = "train", lang = "en"):

		assert config.extension in ["json"] # Only supports csv now

		self.config = config
		self.extension = self.config.extension

		self.max_length = self.config.max_length
		self.max_tweets = self.config.max_tweets
        
		self.lang = lang
		if self.lang == "zh":
 			print("Doing RD for chinese")
 			nlp = nlp_chinese         

		# <------------ Running some defined functions ----------->

		if type_ == "train":
			# self.data_folder_path = self.config.data_folder + "_{}/".format(split)
			self.data_folder_path = self.config.data_folder
			self.train_file_path = self.config.train_file_path
			self.test_1_file_path = self.config.test_1_file_path
			self.test_2_file_path = self.config.test_2_file_path
			self.run_pipeline()

	def get_data(self, type_, return_id = False):

		assert type_ in ["train", "train_test", "test_1", "test_2", "test"]

		max_batch_size = self.config.batch_size if type_ == "train" else self.config.batch_size_test if type_ == "train_test" else self.config.batch_size_test if type_ == "test" else self.config.batch_size_test if type_ == "test_1" else self.config.batch_size_test if type_ == "test_2" else "something is wrong"
		data = self.train_batch if type_ == "train" else self.train_test_batch if type_ == "train_test" else self.test_batch if type_ == "test" else self.test_1_batch if type_ == "test_1" else self.test_2_batch if type_ == "test_2" else "something is wrong"

		for batch in data:

			id_ = getattr(batch, self.config.keys_order["post_id"])

			X = getattr(batch, self.config.keys_order["content"])
			y = getattr(batch, self.config.keys_order["label"])
			structure = getattr(batch, self.config.keys_order["structure"])
			time_delay = getattr(batch, self.config.keys_order["time_delay"])

			# <-------- Getting the sizes --------->
			batch_size, num_articles, num_words, = X.shape

			# <-------- Getting the word_pos tensor --------->
			word_pos = np.repeat(np.expand_dims(np.repeat(np.expand_dims(np.arange(num_words), axis = 0), num_articles, axis = 0), axis  = 0), batch_size, axis = 0)
			word_pos = torch.from_numpy(word_pos)

			# <-------- Getting the attention_mask vector (for words) --------->
			# The mask has 1 for real tokens and 0 for padding / unknown tokens. Only real tokens + last pad are attended to
			# <pad> has an index of 1

			attention_mask_word = torch.where((X == 1), torch.zeros(1), torch.ones(1)).type(torch.FloatTensor)
			check = torch.sum(torch.where((X == 1), torch.ones(1), torch.zeros(1)), dim = -1)

			# <-------- Getting the attention_mask vector (for posts) --------->
			attention_mask_post = torch.where((check == self.config.max_length), torch.zeros(1), torch.ones(1)).type(torch.FloatTensor)

			if batch_size >= len(self.config.gpu_idx):

				if return_id:

					yield id_, X, y, word_pos, time_delay, structure, attention_mask_word, attention_mask_post

				else:

					yield X, y, word_pos, time_delay, structure, attention_mask_word, attention_mask_post

	@staticmethod
	def clean_text(text):

		"""
		This function cleans the text in the following ways:
		1. Replace websites with URL
		1. Replace 's with <space>'s (eg, her's --> her 's)

		"""

		text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "URL", text) # Replace urls with special token
		text = text.replace("\'s", "")
		text = text.replace("\'", "")
		text = text.replace("n\'t", " n\'t")
		text = text.replace("@", "")
		text = text.replace("#", "")
		text = text.replace("_", " ")
		text = text.replace("-", " ")
		text = text.replace("&amp;", "")
		text = text.replace("&gt;", "")
		text = text.replace("\"", "")
		text = text.replace(".", "")
		text = text.replace(",", "")
		text = text.replace("(", "")
		text = text.replace(")", "")

		text = ' '.join(text.split())

		return text.strip()

	@staticmethod
	def clean_tokenized_text(text_lst):

		if len(text_lst) <= 1:
			return text_lst

		idx = 0
		cleaned_token_lst = []

		while idx < len(text_lst) -1:

			current_token = text_lst[idx]
			next_token = text_lst[idx + 1]

			if current_token != next_token:
				cleaned_token_lst.append(current_token)
				idx += 1

			else:

				last_idx = max([i + idx for i, val in enumerate(text_lst[idx:]) if val == current_token]) + 1
				cleaned_token_lst.append(current_token)
				idx = last_idx

		if cleaned_token_lst[-1] != text_lst[-1]:
			cleaned_token_lst.append(text_lst[-1])

		return cleaned_token_lst

	@staticmethod
	def tokenize_structure(structure_lst):

		return structure_lst

	@staticmethod
	def tokenize_text(text):

		text = DataLoader.clean_text(text)
		token_lst = [token.text.lower() for token in nlp(text)]
		token_lst = DataLoader.clean_tokenized_text(token_lst)

		return token_lst

	# Step 1: Define the data fields
	def define_fields(self):

		self.id_field = Field(sequential = False,
							  tokenize = lambda x : x,
							  use_vocab = True)

		self.tweet_field = Field(sequential = True, 
								 tokenize = DataLoader.tokenize_text,
								 include_lengths = False,
								 lower = True,
								 fix_length = self.max_length,
								 use_vocab = True)

		self.timestamp_field = Field(sequential = False, 
									 include_lengths = False,
									 use_vocab = False)

		self.structure_field = Field(sequential = True,
									 tokenize = lambda x : DataLoader.tokenize_structure(x),
									 include_lengths = False,
									 fix_length = self.config.max_tweets,
									 pad_token = self.config.num_structure_index,
									 use_vocab = False)

		self.label_field = Field(sequential = False,
								 use_vocab = False)

		self.tweet_lst_field = NestedField(self.tweet_field,
										   fix_length = self.config.max_tweets)

		self.timestamp_lst_field = NestedField(self.timestamp_field, 
											   pad_token = str(self.config.size),
											   fix_length = self.config.max_tweets)

		self.structure_lst_field = NestedField(self.structure_field,
											   fix_length = self.config.max_tweets)
		
		data_fields = {}

		for key, val in self.config.keys_order.items():

			if key == "post_id":
				data_fields[val] = (val, self.id_field)
			if key == "content":
				data_fields[val] = (val, self.tweet_lst_field)
			elif key == "label":
				data_fields[val] = (val, self.label_field)
			elif key == "time_delay":
				data_fields[val] = (val, self.timestamp_lst_field)
			elif key == "structure":
				data_fields[val] = (val, self.structure_lst_field)

		self.data_fields = data_fields

	# Step 2: Reading the data
	def read_data(self, path):

		data = TabularDataset(path = path,
							  format = self.extension,
							  fields = self.data_fields)

		return data

	# Step 3: Building the vectors
	def build_vectors(self):

		# specify the path to the localy saved vectors (Glove in this case)
		vec = vocab.Vectors(name = self.config.glove_file, cache = self.config.glove_directory)

		self.id_field.build_vocab(getattr(self.train, self.config.keys_order["post_id"]), 
								  getattr(self.test_1, self.config.keys_order["post_id"]), 
								  getattr(self.test_2, self.config.keys_order["post_id"]))

		# Build the vocabulary (for tweets) using the train and test dataset
		self.tweet_field.build_vocab(getattr(self.train, self.config.keys_order["content"]), 
									 getattr(self.test_1, self.config.keys_order["content"]), 
									 getattr(self.test_2, self.config.keys_order["content"]), 
									 max_size = self.config.max_vocab, 
									 vectors = vec)

	# Step 4: Loading the data in batches
	def load_batches(self, dataset, batch_size):

		data = BucketIterator.splits(datasets = (dataset,), # specify data
									 batch_sizes = (batch_size,),  # batch size
									 sort_key = lambda x: len(getattr(x, self.config.keys_order["content"])), # on what attribute the text should be sorted
									 sort_within_batch = True, 
									 repeat = False)

		return data[0]

	def load_vocab_vectors(self, vocab):

		self.tweet_field.vocab = vocab 

	def run_pipeline(self):

		"""
		Pipeline to run all the necessary steps in sequence

		Note: DO NOT CHANGE THE SEQUENCE OF EXECUTION
		"""

		# Step 1 : Define the fields
		self.define_fields()

		# Step 2: Read data
		self.train = self.read_data(os.path.join(self.data_folder_path, self.train_file_path))
		self.test_1 = self.read_data(os.path.join(self.data_folder_path, self.test_1_file_path))
		self.test_2 = self.read_data(os.path.join(self.data_folder_path, self.test_2_file_path))

		# Step 3: Building the vectors 
		self.build_vectors()

		# Step 4: Batching the data
		self.train_batch = self.load_batches(self.train, self.config.batch_size)
		self.train_test_batch = self.load_batches(self.train, self.config.batch_size_test)
		self.test_1_batch = self.load_batches(self.test_1, self.config.batch_size_test)
		self.test_2_batch = self.load_batches(self.test_2, self.config.batch_size_test)

