import torch 
import torch.nn as nn
import torch.nn.functional as F
import os 
import numpy as np

class WordEncoder(nn.Module):

    """

    Encodes the information into vectors

    There are 2 pieces of information that goes into the encoded information: 
    1. Word Embedding 
    2. Position Embedding 

    This set of codes would encode the word embedding information
    
    """

    def __init__(self, config, loader):

        super(WordEncoder, self).__init__()

        self.config = config
        self.loader = loader
        self.vocab = self.loader.tweet_field.vocab 
        self.vocab_vectors = self.loader.tweet_field.vocab.vectors

        # <------------- Defining the word embedding dimensions -------------> 
        self.vocab_size_content = len(self.vocab)
        self.embedding_dim = self.config.emb_dim

        # <------------- Loadings the pretrained embedding weights  -------------> 
        self.emb = nn.Embedding(self.vocab_size_content, self.embedding_dim)
        self.emb.weight.data.copy_(self.vocab_vectors) # load pretrained vectors
        self.emb.weight.requires_grad = self.config.train_word_emb # make embedding non trainable

    def forward(self, src_seq):

        """
        Ref:
        https://pytorch.org/docs/stable/nn.html
        
        Does encoding for the input:
        1. WE encoding 

        <--------- WE Embedding --------->
        Input:
            src_seq : [batch_size (32), max_length_defined (50)]

        Output:
        encoded_we_features : [batch_size (32), max_length_defined (50), we_dimension (300)] tensor

        """
        
        encoded_we_features = self.emb(src_seq)
        
        return encoded_we_features
