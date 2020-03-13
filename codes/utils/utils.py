import torch
import torch.nn as nn
import numpy as np
import pandas as pf
import random
from sklearn.metrics import precision_recall_fscore_support

from config import config
import os
import pickle
os.environ['QT_QPA_PLATFORM']='offscreen'

import matplotlib.pyplot as plt
from collections import OrderedDict

def get_labels_counter(labels):

	labels, counts = np.unique(labels, return_counts = True)
	counter = dict(zip(labels, counts))

	return counter

def get_labels(scores):

	predicted_label = np.argmax(scores, axis = 1).astype(int)

	return predicted_label

def cal_scores(pred_label, true_label, type_ = "all"):

	"""
		Various modes of calculation: 
			- all : calculates for all the samples 
			- (class_label) : calculates for a paticular class
	"""

	# <-------- Getting the relevant data -------->
	true_label_class = true_label
	pred_label_class = pred_label

	if type_ != "all":

		assert type_ in [0,1,2,3] # Must be either of the 4 classes 
		idx = [i for i in range(len(true_label)) if true_label[i] == type_]
		true_label_class = np.array([true_label[i] for i in idx])
		pred_label_class = np.array([pred_label[i] for i in idx])

	assert len(pred_label) == len(true_label)
	assert len(pred_label_class) == len(true_label_class)

	check = pred_label_class == true_label_class
	check = check.astype(int)

	# <-------- Counts -------->
	counter_pred = get_labels_counter(pred_label_class)
	counter_true = get_labels_counter(true_label_class)

	# <-------- Calculations -------->
	acc = np.sum(check) / float(len(pred_label_class))

	if type_ == "all":
		pre, recall, f_score, _  = precision_recall_fscore_support(true_label, pred_label, average = 'macro')
	else:
		pre, recall, f_score, _  = precision_recall_fscore_support(true_label, pred_label, average = 'macro', labels = [type_])

	return acc, pre, recall, f_score,  counter_true, counter_pred


def vis_atten_matrix(dataloader, X, weights, idx = None):

	if idx == None:
		idx = random.randint(0, X.shape[0] -1)

	X_ = X[idx].detach().cpu().numpy()
	w = weights[idx].detach().cpu().numpy()

	content = [dataloader.content_field.vocab.itos[token] for token in X_]

	fig, ax = plt.subplots()
	im = ax.imshow(w, cmap = "hot", interpolation = "nearest")
	ax.set_xticks(np.arange(len(content)))
	ax.set_yticks(np.arange(len(content)))
	ax.set_xticklabels(content)
	ax.set_yticklabels(content)

	fig.colorbar(im)

	plt.setp(ax.get_xticklabels(), rotation = 90, ha = "right", rotation_mode = "anchor")

	plt.show()

def save_vocab_vectors(dataloader, config, path):

	vocab_path = os.path.join(path, "vocab.pkl")
	vectors_path = os.path.join(path, "vectors.pkl")

	with open(vocab_path,'wb') as f:
		pickle.dump(dataloader.tweet_field.vocab, f)

	with open(vectors_path,'wb') as f:
		pickle.dump(dataloader.tweet_field.vocab.vectors, f)

def merge_attention_dict(attention_dict_lst, config, type_):

	"""
		attention_dict_lst is a 4 dim tensor: 

			[batch_size, n_head, len (word/post), len (word/post)]
	"""
	
	dict_keys = list(attention_dict_lst[0].keys())
	new_attention_dict = {}
	for key in dict_keys: 
		new_attention_dict[key] = torch.cat([attention_dict_lst[i][key] for i in range(len(attention_dict_lst))], dim = 0)

	if type_ == "word":
		for key in dict_keys:
			new_attention_dict[key] = new_attention_dict[key].view(-1, config.max_tweets, config.n_head_word, config.max_length, config.max_length)
		
	return new_attention_dict

def merge_dict(dict_1, dict_2):

	new_dict = {}

	keys = list(dict_1.keys()) + list(dict_2.keys())

	for key in keys:

		count = 0 
		if key in list(dict_1.keys()):
			count += dict_1[key]

		if key in list(dict_2.keys()):
			count += dict_2[key]

		new_dict[key] = count

	return new_dict

def create_new_state_dict(current_state_dict):

	new_state_dict = OrderedDict()
	for k, v in current_state_dict.items():
		name = k[7:] # remove module.
		new_state_dict[name] = v

	return new_state_dict

def make_dir(path):

	saved_models_path = os.path.join(path, "saved_notes")
	check_points_path = os.path.join(path, "check_point")
	best_model_path = os.path.join(path, "best_model")

	paths = [path, saved_models_path, check_points_path, best_model_path]
	
	for p in paths:
		try:
			os.mkdir(p)
		except OSError:
			print("Creation of path {} failed".format(p))


def check_point_epoch(epoch, model, word_encoder, word_pos_encoder, time_delay_encoder, optimizer, loss, 
					  acc_train, precision_train, recall_train, f_score_train, counter_true_train, counter_pred_train, 
					  acc_test_1, precision_test_1, recall_test_1, f_score_test_1, counter_true_test_1, counter_pred_test_1,
					  acc_test_2, precision_test_2, recall_test_2, f_score_test_2, counter_true_test_2, counter_pred_test_2, 
					  acc_test, precision_test, recall_test, f_score_test, counter_true_test, counter_pred_test, 
					  path):

	path = os.path.join(path, "check_point", "epoch" + str(epoch) + ".tar")

	torch.save({
				"epoch" : epoch,
				"model_state_dict" : model.state_dict(),
				"word_encoder" : word_encoder.state_dict(),
				"word_pos_encoder" : word_pos_encoder.state_dict(),
				"time_delay_encoder" : time_delay_encoder.state_dict(),
				"optimizer_state_dict" : optimizer.state_dict(),
				"loss" : loss,
				"acc_train" : acc_train,
				"precision_train" : precision_train,
				"recall_train" : recall_train, 
				"f_score_train" : f_score_train,
				"counter_true_train" : counter_true_train,
				"counter_pred_train" : counter_pred_train,
				"acc_test_1" : acc_test_1,
				"precision_test_1" : precision_test_1,
				"recall_test_1" : recall_test_1, 
				"f_score_test_1" : f_score_test_1,
				"counter_true_test_1" : counter_true_test_1,
				"counter_pred_test_1" : counter_pred_test_1,
				"acc_test_2" : acc_test_2,
				"precision_test_2" : precision_test_2,
				"recall_test_2" : recall_test_2, 
				"f_score_test_2" : f_score_test_2,
				"counter_true_test_2" : counter_true_test_2,
				"counter_pred_test_2" : counter_pred_test_2,
				"acc_test" : acc_test,
				"precision_test" : precision_test,
				"recall_test" : recall_test, 
				"f_score_test" : f_score_test,
				"counter_true_test" : counter_true_test,
				"counter_pred_test" : counter_pred_test,
				}, path)


def log_info(path, msg):
	file = os.path.join(path, "logs")

	with open(file, "a+", encoding = "UTF-8") as f:

		if type(msg) == str:
			f.write(msg)
			f.write("\n")

		if type(msg) == dict:
			for key, val in msg.items():
				f.write(str(key) + " : " + str(val))
				f.write("\n")

def log_best_model_info(path, epoch, msg, type_, file):

	file = os.path.join(path, "saved_notes", "best_model_log_" + type_ + "_" + file)

	with open(file, "w", encoding = "UTF-8") as f:

		f.write(epoch)
		f.write("\n")

		for key, val in msg.items():
			f.write(str(key) + " : " + str(val))
			f.write("\n")

def save_best_model(path, model, word_encoder, word_pos_encoder, time_delay_encoder, optimizer, type_, file):

	path_model = os.path.join(path, "best_model", "best_model_" + type_ + "_" + file + ".pt")
	path_word_encoder = os.path.join(path, "best_model","best_model_word_encoder_" + type_ + "_" + file + ".pt")
	path_word_pos_encoder = os.path.join(path, "best_model", "best_model_word_pos_encoder_" + type_ + "_" + file + ".pt")
	path_time_delay_encoder = os.path.join(path, "best_model", "best_model_time_delay_encoder_" + type_ + "_" + file + ".pt")
	path_optimizer = os.path.join(path, "best_model", "best_model_optimizer_" + type_ + "_" + file + ".pt")
	
	torch.save(model.state_dict(), path_model)
	torch.save(word_encoder.state_dict(), path_word_encoder)
	torch.save(word_pos_encoder.state_dict(), path_word_pos_encoder)
	torch.save(time_delay_encoder.state_dict(), path_time_delay_encoder)
	torch.save(optimizer.state_dict(), path_optimizer)


def plot_graphs(path, epoch_values):

	epoch_lst = []
	
	train_loss_lst = []
	
	train_acc_lst = []
	train_f1_lst = []

	test_1_acc_lst = []
	test_1_f1_lst = []

	test_2_acc_lst = []
	test_2_f1_lst = []

	test_acc_lst = []
	test_f1_lst = []
	
	loss_file = os.path.join(path, "saved_notes", "loss_graph.png")

	train_acc_file = os.path.join(path, "saved_notes", "train_accuracy_graph.png")
	train_f1_file = os.path.join(path, "saved_notes", "train_f1_graph.png")
	
	test_1_acc_file = os.path.join(path, "saved_notes", "test_1_accuracy_graph.png")
	test_1_f1_file = os.path.join(path, "saved_notes", "test_1_f1_graph.png")

	test_2_acc_file = os.path.join(path, "saved_notes", "test_2_accuracy_graph.png")
	test_2_f1_file = os.path.join(path, "saved_notes", "test_2_f1_graph.png")

	test_acc_file = os.path.join(path, "saved_notes", "test_accuracy_graph.png")
	test_f1_file = os.path.join(path, "saved_notes", "test_f1_graph.png")

	for epoch, record in epoch_values.items():
		
		epoch_lst.append(epoch)
		train_loss_lst.append(record["loss"])

		train_acc_lst.append(record["acc_train"])
		train_f1_lst.append(record["f_score_train"])

		test_1_acc_lst.append(record["acc_test_1"])
		test_1_f1_lst.append(record["f_score_test_1"])

		test_2_acc_lst.append(record["acc_test_2"])
		test_2_f1_lst.append(record["f_score_test_2"])

		test_acc_lst.append(record["acc_test"])
		test_f1_lst.append(record["f_score_test"])

	plt.plot(epoch_lst, train_loss_lst)
	plt.title("Training loss VS epoch")
	plt.xlabel("epoch")
	plt.ylabel("training_loss")
	plt.savefig(loss_file)
	#plt.show()
	plt.gcf().clear()

	plt.plot(epoch_lst, train_acc_lst)
	plt.title("Accuracy (Train) VS epoch")
	plt.xlabel("epoch")
	plt.ylabel("train_acc")
	plt.savefig(train_acc_file)
	#plt.show()
	plt.gcf().clear()

	plt.plot(epoch_lst, train_f1_lst)
	plt.title("F1 (Train) VS epoch")
	plt.xlabel("epoch")
	plt.ylabel("train_f1")
	plt.savefig(train_f1_file)
	#plt.show()
	plt.gcf().clear()

	plt.plot(epoch_lst, test_1_acc_lst)
	plt.title("Accuracy (Test 1) VS epoch")
	plt.xlabel("epoch")
	plt.ylabel("test_1_acc")
	plt.savefig(test_1_acc_file)
	#plt.show()
	plt.gcf().clear()

	plt.plot(epoch_lst, test_1_f1_lst)
	plt.title("F1 (Test 1) VS epoch")
	plt.xlabel("epoch")
	plt.ylabel("test_1_f1")
	plt.savefig(test_1_f1_file)
	#plt.show()
	plt.gcf().clear()

	plt.plot(epoch_lst, test_2_acc_lst)
	plt.title("Accuracy (Test 2) VS epoch")
	plt.xlabel("epoch")
	plt.ylabel("test_2_acc")
	plt.savefig(test_2_acc_file)
	#plt.show()
	plt.gcf().clear()

	plt.plot(epoch_lst, test_2_f1_lst)
	plt.title("F1 (Test 2) VS epoch")
	plt.xlabel("epoch")
	plt.ylabel("test_2_f1")
	plt.savefig(test_2_f1_file)
	#plt.show()
	plt.gcf().clear()

	plt.plot(epoch_lst, test_acc_lst)
	plt.title("Accuracy (Test) VS epoch")
	plt.xlabel("epoch")
	plt.ylabel("test_acc")
	plt.savefig(test_acc_file)
	#plt.show()
	plt.gcf().clear()

	plt.plot(epoch_lst, test_f1_lst)
	plt.title("F1 (Test) VS epoch")
	plt.xlabel("epoch")
	plt.ylabel("test_f1")
	plt.savefig(test_f1_file)
	#plt.show()
	plt.gcf().clear()