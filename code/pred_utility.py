from utility  import load_pickle, read, write, encode_to_id, prep_to_class, id_to_label, reconstruct_pred


def load_hp():
	"""
	Load hyperparameters.

	params:
	    Model (str)

	returns: int, int, int, int
	"""
	MAX_LENGTH         = 80
	UNI_EMBEDDING_SIZE = 32
	BI_EMBEDDING_SIZE  = 32
	HIDDEN_SIZE        = 128

	return MAX_LENGTH, UNI_EMBEDDING_SIZE, BI_EMBEDDING_SIZE, HIDDEN_SIZE


def pred_helper(input_path, resources_path, MAX_LENGTH):

	"""
	Prepares for the prediction.

	params:
		input_path (str)
		resources_path (str)

	returns: int, int, int, int
	"""

	input_pred      = read(input_path)
	uni_dict_file   = resources_path+"/Dicts/uni_to_id.pickle"
	bi_dict_file    = resources_path+"/Dicts/bi_to_id.pickle"

	uni_dict        = load_pickle(uni_dict_file)
	bi_dict         = load_pickle(bi_dict_file)

	UNI_VOCAB_SIZE  = len(uni_dict) #Length of unigram dictionary
	BI_VOCAB_SIZE   = len(bi_dict)  #Length of bigram dictionary
	  
	orig_len, uni_train_set   = prep_to_class(input_pred, uni_dict, MAX_LENGTH, False)
	_,        bi_train_set    = prep_to_class(input_pred, bi_dict, MAX_LENGTH, True)


	return UNI_VOCAB_SIZE,  BI_VOCAB_SIZE, uni_train_set, bi_train_set, orig_len


def write_pred(pred, seq_len, output_path):
	"""
	write prediction txt.

	params:
		pred (np array)
		seq_len (np array)
		output_path (str)
	"""
	label_dict = id_to_label()
	re_pred = reconstruct_pred(pred, seq_len)
	with open(output_path, 'w') as f:
		for line in re_pred:
			for char in line:
				f.write(label_dict[char])
			f.write('\n')


def load_ckpt(resources_path):
	"""
	loads checkpoint path for best model.

	params:
	    resources_path (str)
	returns: str
	"""
	ckpt = resources_path+"/Best_Model"
	return ckpt


