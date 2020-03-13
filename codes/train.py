import os
from config import config
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in vars(config)["gpu_idx"])

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from tqdm import tqdm 
from datetime import datetime

# <------- GPU optimization code -------->
from utils.parallel import DataParallelModel, DataParallelCriterion

# <------- Self defined classes -------->
from DataLoader import DataLoader
from Transformer import HierarchicalTransformer
from Encoder import WordEncoder
from Encoder import PositionEncoder
from Optimizer import Optimizer

from utils.utils import * 

__author__ = "Serena Khoo"

class Trainer():

	def __init__(self, dataloader, hierarchical_transformer, config, i):

		super(Trainer, self).__init__()

		self.iter = i
		self.config = config
		self.cpu = torch.device("cpu")
		self.multi_gpu = len(self.config.gpu_idx) > 1

		self.dataloader = dataloader
		self.word_encoder = WordEncoder.WordEncoder(config, self.dataloader.tweet_field.vocab)
		self.word_pos_encoder = PositionEncoder.PositionEncoder(config, self.config.max_length)
		self.time_delay_encoder = PositionEncoder.PositionEncoder(config, self.config.size)

		# <----------- Check for GPU setting ----------->
		if self.config.gpu:

			self.hierarchical_transformer = DataParallelModel(hierarchical_transformer.cuda())
			self.criterion = DataParallelCriterion(nn.NLLLoss())

		else:
			self.hierarchical_transformer = hierarchical_transformer
			self.criterion = nn.NLLLoss()

		self.adam_optimizer = optim.Adam(self.hierarchical_transformer.parameters(), np.power(self.config.d_model, - 0.5), betas = (self.config.beta_1, self.config.beta_2))
		self.optimizer = Optimizer.Optimizer(self.config, self.adam_optimizer)
	
	def test_performance(self, type_):

		predicted_y_lst = []
		y_lst = []

		self.hierarchical_transformer.eval() # Make sure that it is on eval mode first

		with torch.no_grad():

			for X, y, word_pos, time_delay, structure, attention_mask_word, attention_mask_post in self.dataloader.get_data(type_):

				# <-------- Casting as a variable --------->
				X = Variable(X)
				y = Variable(y)
				word_pos = Variable(word_pos)
				time_delay = Variable(time_delay)
				structure = Variable(structure)
				attention_mask_word = Variable(attention_mask_word)
				attention_mask_post = Variable(attention_mask_post)

				# <-------- Encode content -------------->
				X = self.word_encoder(X)
				word_pos = self.word_pos_encoder(word_pos)
				time_delay = self.time_delay_encoder(time_delay)

				# <-------- Move to GPU -------------->
				if self.config.gpu:
					X = X.cuda()
					y = y.cuda()
					word_pos = word_pos.cuda()
					time_delay = time_delay.cuda()
					structure = structure.cuda()
					attention_mask_word = attention_mask_word.cuda()
					attention_mask_post = attention_mask_post.cuda()

				# <--------- Getting the predictions ---------> 
				predicted_y = self.hierarchical_transformer(X, word_pos, time_delay, structure, attention_mask_word = attention_mask_word, attention_mask_post = attention_mask_post)

				# predicted_y, self_atten_output_post, self_atten_weights_dict_word, self_atten_weights_dict_post = self.hierarchical_transformer(X, word_pos, time_delay)
				# self_atten_weights_dict_word = merge_attention_dict(self_atten_weights_dict_word, self.config, "word")
				# self_atten_weights_dict_post = merge_attention_dict(self_atten_weights_dict_post, self.config, "post")

				if self.multi_gpu:
					predicted_y = torch.cat(list(predicted_y), dim = 0)

				# <------- to np array ------->
				predicted_y = predicted_y.cpu().numpy()
				y = y.cpu().numpy()

				print("test", predicted_y)

				# <------- Appending it to the master list ------->
				predicted_y_lst.extend(predicted_y)
				y_lst.extend(y)

				# <--------- Free up the GPU -------------->
				del X
				del y
				del predicted_y
				del word_pos
				del time_delay
				del structure

			# <------- Get scores ------->
			predicted_y_lst = np.array(predicted_y_lst)
			predicted_y_lst = get_labels(predicted_y_lst)
			y_lst = np.array(y_lst)

			return predicted_y_lst, y_lst

	def train(self):

		print("*" * 40 + " START OF TRAINING " + "*" * 40)

		epoch_values = {}

		# <------ Gets for test 1 ------>
		best_acc_test_1 = 0.0
		best_f_score_test_1 = 0.0
		best_acc_test_1_for_2 = 0.0
		best_f_score_test_1_for_2 = 0.0

		best_record_f_score_test_1 = {}
		best_record_accuracy_test_1 = {}

		# <------ Gets for test 2 ------>
		best_acc_test_2 = 0.0
		best_f_score_test_2 = 0.0
		best_acc_test_2_for_1 = 0.0
		best_f_score_test_2_for_1 = 0.0

		best_record_f_score_test_2 = {}
		best_record_accuracy_test_2 = {}

		# <------ Gets for full test ------>
		best_acc_test = 0.0
		best_f_score_test = 0.0

		best_record_f_score_test = {}
		best_record_accuracy_test = {}

		# <------ For logging purpose ------>

		dataset = self.config.data_folder.split("/")[-1]
		name = "{}_split_{}_{}".format(dataset, self.iter, datetime.now().strftime('%Y-%m-%d-%H:%M:%S')) # Date & Time for logging purposes
		path = os.path.join(self.config.log_folder, self.config.dataset_name, name + "_" + self.config.experiment_name)

		make_dir(path)
		print(path)
		save_vocab_vectors(self.dataloader, self.config, path)
		log_info(path, "*" * 40 + " EXPERIMENT " + "*" * 40)
		log_info(path, "*" * 40 + " SPLIT {} ".format(self.iter) + "*" * 40)
		log_info(path, str(vars(self.config)))
		log_info(path, "*" * 90)

		for epoch in tqdm(range(self.config.num_epoch)):

			running_loss = 0
			i  = 0

			for X, y, word_pos, time_delay, structure, attention_mask_word, attention_mask_post in self.dataloader.get_data("train"):

				# <-------- Casting as a variable --------->
				X = Variable(X)
				y = Variable(y)
				word_pos = Variable(word_pos)
				time_delay = Variable(time_delay)
				structure = Variable(structure)
				attention_mask_word = Variable(attention_mask_word)
				attention_mask_post = Variable(attention_mask_post)
				
				# <-------- Encode content -------------->
				X = self.word_encoder(X)
				word_pos = self.word_pos_encoder(word_pos)
				time_delay = self.time_delay_encoder(time_delay)

				# <-------- Move to GPU -------------->
				if self.config.gpu:
					X = X.cuda()
					y = y.cuda()
					word_pos = word_pos.cuda()
					time_delay = time_delay.cuda()
					structure = structure.cuda()
					attention_mask_word = attention_mask_word.cuda()
					attention_mask_post = attention_mask_post.cuda()
				
				# <------- Settings ------------->
				self.hierarchical_transformer.train() # Set the model to be on train mode (So that the dropout applies)
				self.optimizer.zero_grad() # zero grad it

				# <--------- Getting the predictions ---------> 
				predicted_y = self.hierarchical_transformer(X, word_pos, time_delay, structure, attention_mask_word = attention_mask_word, attention_mask_post = attention_mask_post)
				
				#predicted_y, self_atten_output_post, self_atten_weights_dict_word, self_atten_weights_dict_post = self.hierarchical_transformer(X, word_pos, time_delay)
				# self_atten_weights_dict_word = merge_attention_dict(self_atten_weights_dict_word, self.config, "word")
				# self_atten_weights_dict_post = merge_attention_dict(self_atten_weights_dict_post, self.config, "post")

				print(predicted_y)

				# <--------- Getting loss and backprop --------->
				loss = self.criterion(predicted_y, y)
				loss.backward()
				self.optimizer.step_and_update_lr()

				# <--------- Calculating the loss --------->
				running_loss += float(loss.detach().item())
				i += 1 

				# <--------- Free up the GPU -------------->
				del X
				del y
				del predicted_y
				del word_pos
				del time_delay
				del structure
			
				torch.cuda.empty_cache()

			record = {}
			running_loss = running_loss/ float(i)
			print()
			print("Epoch {}: {}".format(epoch + 1, running_loss))

			with torch.no_grad():

				pred_train, true_train = self.test_performance("train_test")
				pred_test_1,true_test_1 = self.test_performance("test_1")
				pred_test_2,true_test_2 = self.test_performance("test_2")

				pred_test = np.concatenate((pred_test_1, pred_test_2))
				true_test = np.concatenate((true_test_1, true_test_2))

				# <-------- Getting performance for all the clases  -------->
				acc_train, pre_train, recall_train, f_score_train, counter_true_train, counter_pred_train = cal_scores(pred_train, true_train, type_ = "all") 
				acc_test_1, pre_test_1, recall_test_1, f_score_test_1, counter_true_test_1, counter_pred_test_1 = cal_scores(pred_test_1, true_test_1, type_ = "all") 
				acc_test_2, pre_test_2, recall_test_2, f_score_test_2, counter_true_test_2, counter_pred_test_2 = cal_scores(pred_test_2, true_test_2, type_ = "all") 
				acc_test, pre_test, recall_test, f_score_test, counter_true_test, counter_pred_test = cal_scores(pred_test, true_test, type_ = "all") 

				# <-------- Getting performance for individual claseses  -------->
				acc_test_1_class_0, pre_test_1_class_0, recall_test_1_class_0, f_score_test_1_class_0, counter_true_test_1_class_0, counter_pred_test_1_class_0 = cal_scores(pred_test_1, true_test_1, type_ = 0) 
				acc_test_1_class_1, pre_test_1_class_1, recall_test_1_class_1, f_score_test_1_class_1, counter_true_test_1_class_1, counter_pred_test_1_class_1 = cal_scores(pred_test_1, true_test_1, type_ = 1)

				acc_test_2_class_0, pre_test_2_class_0, recall_test_2_class_0, f_score_test_2_class_0, counter_true_test_2_class_0, counter_pred_test_2_class_0 = cal_scores(pred_test_2, true_test_2, type_ = 0) 
				acc_test_2_class_1, pre_test_2_class_1, recall_test_2_class_1, f_score_test_2_class_1, counter_true_test_2_class_1, counter_pred_test_2_class_1 = cal_scores(pred_test_2, true_test_2, type_ = 1)

				acc_test_class_0, pre_test_class_0, recall_test_class_0, f_score_test_class_0, counter_true_test_class_0, counter_pred_test_class_0 = cal_scores(pred_test, true_test, type_ = 0) 
				acc_test_class_1, pre_test_class_1, recall_test_class_1, f_score_test_class_1, counter_true_test_class_1, counter_pred_test_class_1 = cal_scores(pred_test, true_test, type_ = 1)

				if epoch%self.config.interval == 0:

					check_point_epoch(epoch + 1, 
										self.hierarchical_transformer,
										self.word_encoder,
										self.word_pos_encoder,
										self.time_delay_encoder,
										self.optimizer,
										running_loss,
										acc_train,
										pre_train,
										recall_train,
										f_score_train,
										counter_true_train,
										counter_pred_train,
										acc_test_1,
										pre_test_1,
										recall_test_1,
										f_score_test_1,
										counter_true_test_1,
										counter_pred_test_1,
										acc_test_2,
										pre_test_2,
										recall_test_2,
										f_score_test_2,
										counter_true_test_2,
										counter_pred_test_2,
										acc_test,
										pre_test,
										recall_test,
										f_score_test,
										counter_true_test,
										counter_pred_test,
										path)
 
				record["epoch"] = epoch + 1
				record["loss"] = running_loss

				record["acc_train"] = acc_train
				record["precision_train"] = pre_train
				record["recall_train"] = recall_train
				record["f_score_train"] = f_score_train
				record["counter_true_train"] = counter_true_train
				record["counter_pred_train"] = counter_pred_train

				record["acc_test_1"] = acc_test_1
				record["precision_test_1"] = pre_test_1
				record["recall_test_1"] = recall_test_1
				record["f_score_test_1"] = f_score_test_1
				record["counter_true_test_1"] = counter_true_test_1
				record["counter_pred_test_1"] = counter_pred_test_1

				record["acc_test_2"] = acc_test_2
				record["precision_test_2"] = pre_test_2
				record["recall_test_2"] = recall_test_2
				record["f_score_test_2"] = f_score_test_2
				record["counter_true_test_2"] = counter_true_test_2
				record["counter_pred_test_2"] = counter_pred_test_2

				record["acc_test"] = acc_test
				record["precision_test"] = pre_test
				record["recall_test"] = recall_test
				record["f_score_test"] = f_score_test
				record["counter_true_test"] = counter_true_test
				record["counter_pred_test"] = counter_pred_test

				# <--------- test 1 --------->
				record["acc_test_1_classes"] = {0 : acc_test_1_class_0,
												1 : acc_test_1_class_1}

				record["precision_test_1_classes"] = {0 : pre_test_1_class_0,
													  1 : pre_test_1_class_1}

				record["recall_test_1_classes"] = {0 : recall_test_1_class_0,
									  			   1 : recall_test_1_class_1}
				
				record["f_score_test_1_classes"] = {0 : f_score_test_1_class_0,
									  			    1 : f_score_test_1_class_1}

				record["counter_pred_test_1_classes"] = {0 : counter_pred_test_1_class_0,
														 1 : counter_pred_test_1_class_1}

				# <--------- test 2 --------->
				record["acc_test_2_classes"] = {0 : acc_test_2_class_0,
												1 : acc_test_2_class_1}

				record["precision_test_2_classes"] = {0 : pre_test_2_class_0,
													  1 : pre_test_2_class_1}

				record["recall_test_2_classes"] = {0 : recall_test_2_class_0,
									  			   1 : recall_test_2_class_1}

				record["f_score_test_2_classes"] = {0 : f_score_test_2_class_0,
									  			    1 : f_score_test_2_class_1}

				record["counter_pred_test_2_classes"] = {0 : counter_pred_test_2_class_0,
														 1 : counter_pred_test_2_class_1}

				# <--------- test --------->
				record["acc_test_classes"] = {0 : acc_test_class_0,
											  1 : acc_test_class_1}

				record["precision_test_classes"] = {0 : pre_test_class_0,
													1 : pre_test_class_1}

				record["recall_test_classes"] = {0 : recall_test_class_0,
												 1 : recall_test_class_1}
				
				record["f_score_test_classes"] = {0 : f_score_test_class_0,
									  			   1 : f_score_test_class_1}

				record["counter_pred_test_classes"] = {0 : counter_pred_test_class_0,
													   1 : counter_pred_test_class_1}

				epoch_values[epoch + 1] = record

				log_info(path, record)
				log_info(path, "=" * 90)

				if f_score_test_1 >= best_f_score_test_1:

					print("CURRENT BEST (F-SCORE) FOUND AT EPOCH : {} (EVALUATED WITH {})".format(epoch + 1, "TEST_1"))

					if f_score_test_1 == best_f_score_test_1:

						if f_score_test_2 >= best_f_score_test_1_for_2:

							best_f_score_test_1_for_2 = f_score_test_2
							best_f_score_test_1 = f_score_test_1
							best_record_f_score_test_1 = record

							log_best_model_info(path, "epoch : " + str(epoch + 1), best_record_f_score_test_1, "f_score", "test_1")
							save_best_model(path, self.hierarchical_transformer, self.word_encoder, self.word_pos_encoder, self.time_delay_encoder, self.optimizer, "f_score", "test_1")

					else:

						best_f_score_test_1 = f_score_test_1
						best_record_f_score_test_1 = record
						best_f_score_test_1_for_2 = f_score_test_2

						log_best_model_info(path, "epoch : " + str(epoch + 1), best_record_f_score_test_1, "f_score", "test_1")
						save_best_model(path, self.hierarchical_transformer, self.word_encoder, self.word_pos_encoder, self.time_delay_encoder, self.optimizer, "f_score", "test_1")

				if acc_test_1 >= best_acc_test_1:

					print("CURRENT BEST (ACCURACY) FOUND AT EPOCH : {} (EVALUATED WITH {})".format(epoch + 1, "TEST_1"))
					
					if acc_test_1 == best_acc_test_1:

						if acc_test_2 >= best_acc_test_1_for_2:

							best_acc_test_1_for_2 = acc_test_2
							best_acc_test_1 = acc_test_1
							best_record_acc_test_1 = record

							log_best_model_info(path, "epoch : " + str(epoch + 1), best_record_acc_test_1, "accuracy", "test_1")
							save_best_model(path, self.hierarchical_transformer, self.word_encoder, self.word_pos_encoder, self.time_delay_encoder, self.optimizer, "accuracy", "test_1")

					else:

						best_acc_test_1 = acc_test_1
						best_record_acc_test_1 = record
						best_acc_test_1_for_2 = acc_test_2

						log_best_model_info(path, "epoch : " + str(epoch + 1), best_record_acc_test_1, "accuracy", "test_1")
						save_best_model(path, self.hierarchical_transformer, self.word_encoder, self.word_pos_encoder, self.time_delay_encoder, self.optimizer, "accuracy", "test_1")

				if f_score_test_2 >= best_f_score_test_2:

					print("CURRENT BEST (F-SCORE) FOUND AT EPOCH : {} (EVALUATED WITH {})".format(epoch + 1, "TEST_2"))

					if f_score_test_2 == best_f_score_test_2:

						if f_score_test_1 >= best_f_score_test_2_for_1:

							best_f_score_test_2_for_1 = f_score_test_1
							best_f_score_test_2 = f_score_test_2
							best_record_f_score_test_2 = record

							log_best_model_info(path, "epoch : " + str(epoch + 1), best_record_f_score_test_2, "f_score", "test_2")
							save_best_model(path, self.hierarchical_transformer, self.word_encoder, self.word_pos_encoder, self.time_delay_encoder, self.optimizer, "f_score", "test_2")

					else:

						best_f_score_test_2 = f_score_test_2
						best_record_f_score_test_2 = record
						best_f_score_test_2_for_1 = f_score_test_1

						log_best_model_info(path, "epoch : " + str(epoch + 1), best_record_f_score_test_1, "f_score", "test_1")
						save_best_model(path, self.hierarchical_transformer, self.word_encoder, self.word_pos_encoder, self.time_delay_encoder, self.optimizer, "f_score", "test_1")

				if acc_test_2 >= best_acc_test_2:
					print("CURRENT BEST (ACCURACY) FOUND AT EPOCH : {} (EVALUATED WITH {})".format(epoch + 1, "TEST_2"))
					
					if acc_test_2 == best_acc_test_2:

						if acc_test_1 >= best_acc_test_2_for_1:

							best_acc_test_2_for_1 = acc_test_1
							best_acc_test_2 = acc_test_2
							best_record_acc_test_2 = record

							log_best_model_info(path, "epoch : " + str(epoch + 1), best_record_acc_test_2, "accuracy", "test_2")
							save_best_model(path, self.hierarchical_transformer, self.word_encoder, self.word_pos_encoder, self.time_delay_encoder, self.optimizer, "accuracy", "test_2")

					else:

						best_acc_test_2 = acc_test_2
						best_record_acc_test_2 = record
						best_acc_test_2_for_1 = acc_test_1

						log_best_model_info(path, "epoch : " + str(epoch + 1), best_record_acc_test_2, "accuracy", "test_2")
						save_best_model(path, self.hierarchical_transformer, self.word_encoder, self.word_pos_encoder, self.time_delay_encoder, self.optimizer, "accuracy", "test_2")

				if f_score_test >= best_f_score_test:
					print("CURRENT BEST (F-SCORE) FOUND AT EPOCH : {} (EVALUATED WITH {})".format(epoch + 1, "TEST"))
					
					best_f_score_test = f_score_test
					best_record_f_score_test = record

					log_best_model_info(path, "epoch : " + str(epoch + 1), best_record_f_score_test, "f_score", "test")
					save_best_model(path, self.hierarchical_transformer, self.word_encoder, self.word_pos_encoder, self.time_delay_encoder, self.optimizer, "f_score", "test")

				if acc_test >= best_acc_test:
					print("CURRENT BEST (ACCURACY) FOUND AT EPOCH : {} (EVALUATED WITH {})".format(epoch + 1, "TEST"))
					
					best_acc_test = acc_test
					best_record_acc_test = record

					log_best_model_info(path, "epoch : " + str(epoch + 1), best_record_acc_test, "accuracy", "test")
					save_best_model(path, self.hierarchical_transformer, self.word_encoder, self.word_pos_encoder, self.time_delay_encoder, self.optimizer, "accuracy", "test")

		plot_graphs(path, epoch_values)
		print("*" * 40 + " DONE WITH TRAINING " + "*" * 40)


if __name__ == "__main__":

	for i in range(1):
		
		print("Training for split {}".format(i))

		loader = DataLoader.DataLoader(config, i)
		hierarchical_transformer = HierarchicalTransformer.HierarchicalTransformer(config)
		trainer = Trainer(loader, hierarchical_transformer, config, i)
		trainer.train()

		del loader
		del hierarchical_transformer
		del trainer

