# Description:
	This is the source code for Interpretable Rumor Detection in Microblogs by Attending to User Interactions by Ling Min Serena Khoo, Hai Leong Chieu, Zhong Qian, Jing Jiang. Published in Thirty-Fourth AAAI Conference on Artificial Intelligence 2020.
	Archive: https://arxiv.org/abs/2001.10667

# Dependencies
	This set of code was developed in python3 and pytorch. The packages and their respective versions are state in requirements.txt
	To install the dependencies, run pip install -r requirements.txt

# Training of model
	The code for training the model is found in : ./codes
		- The python script that would be used to train the models would be ./train.py
		- Change the config file (./config.py) to change the parameters before training
		- The config file would allow you to change the path of the train, test and val data.
		- The config file would allow you to change the GPUs that would be used for training.
		- Parameters include:
		a. include_key_structure: Boolean, indicating if we should perform structure-aware attention
		b. include_val_structure: Boolean, indicating if we should propagate structure information in the MHA layers
		c. word_module_version: Either 0,1,2,3 or 4, {0: max_pooling, 1: average_pooling, 2: max_pooling_w_attention, 3: average_pooling_w_attention, 4: attention}. The version used in PLAN and STAPLAN is 2, HiT-STAPLAN is 4. 
		d. post_module_version: Either 0,1,2 or 3, {0: average_pooling, 1: condense_into_fix_vector, 2: first_vector, 3: attention}. The version used in all 3 models in the paper is 3. 
		e. vary_LR: Boolean, indicating if we should use a triangular learning rate (Increases initially then decreases after a certain point)
		f. ff_word: Boolean, indicating if we should finetune the token embedding
		g. ff_post: Boolean, indicating if we should finetune the post embedding
		- To run the training code, run the command "python train.py"

# Testing model
	The code for testing the model is found in : ./codes
	- The ipython notebook that would be used to test the models would be ./testing_script.ipynb
	- Change the config file (./config.py) to change the parameters before testing
	- The config file would allow you to change the following:
	    - The model to select 
	        - Models to be selected could be found in ./logs after training
	- The path of the testing data
	- The path of the output from testing
 

# Contact:
	Please contact Serena (klingmin@dso.org.sg) for any enquiries. Thank you!