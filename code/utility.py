import pickle
import numpy as np

def read(file_path, encoding='utf8'):
	"""
	Read input file.

	params:
	    file_path (str)

	returns: str
	"""
	with open(file_path, 'r') as f:
		return f.read()


def write(content, file_path, encoding='utf8'):
    """
    Write to output file

    params:
        content List[List[str]]
        file_path (str)
        encoding (str) [default='utf8']
    """
    with open(file_path, 'w', encoding=encoding) as f:
        for line in content:
            f.write(''.join(line))
            if line != content[-1]:
                f.write('\n')


def load_pickle(pickle_file):
	"""
	Pickle dictionary.

	params:
	    pickle_file (pickle)
	"""
	with open(pickle_file, 'rb') as h:
		return pickle.load(h)


def encode_to_id(file, dictionary, max_length, bi):
	"""
	Prepares numpy array of indices according to the dictionary.

	params:
		file (str)
		dictionary (dict)
		max_length (int)
		bi (boolean)
	returns: numpy array
	"""
	number_of_ex = len(file.splitlines())
	vocab_size   = len(dictionary) #dictionary size
	encoded      = np.zeros([number_of_ex, max_length])
	for i, line in enumerate(file.splitlines()):
		line_list = list(line)

		if bi:
			line_list.append("</S>")

		for j, word in enumerate(line_list) :
			if j > max_length-1 or word == "</S>":
				break
			if bi:
				gram = word + line_list[j+1]
			else:
				gram = word
			
			if gram in dictionary:   
				encoded[i,j] = dictionary[gram]
			elif gram != "\n":
				encoded[i,j] = dictionary["<UNK>"]

	return encoded


def prep_to_class(file, dictionary, max_length, bi):
	"""
	Prepare input for classification.

	params:
		file (str)
		dictionary (dict)
		max_length (int)
		bi (boolean)
	returns: List, numpy array
	"""
	orig_len = []
	for i in file.splitlines():
		orig_len.append(len(i))

	trunc_list = file.splitlines()
	for i, val in enumerate(trunc_list):
		if len(val)>max_length: 
			trunc_list.insert(i+1, val[max_length:])
			trunc_list[i]=val[:max_length]
 

	number_of_ex = len(trunc_list)
	vocab_size   = len(dictionary) #dictionary size
	encoded      = np.zeros([number_of_ex, max_length])
	for i, line in enumerate(trunc_list):
		line_list = list(line)
		if bi:
			line_list.append("</S>")
		for j, word in enumerate(line_list) :
			if j > max_length-1 or word == "</S>":
				break
			if bi:
				gram = word + line_list[j+1]
			else:
				gram = word
			
			if gram in dictionary:   
				encoded[i,j] = dictionary[gram]
			elif gram != "\n":
				encoded[i,j] = dictionary["<UNK>"]
	return orig_len, encoded


def id_to_label():
    """
    Get label to id dictionary.

    params:

    returns: dict
    """
    id_to_label = dict()
    id_to_label[0] = "B"
    id_to_label[1] = "E" 
    id_to_label[2] = "I" 
    id_to_label[3] = "S"

    return id_to_label



def reconstruct_pred(pred, seq_len):
	"""
	Reconstructs labels - predictions.

	params:
		pred (np array)
		seq_len (np array)

    returns: List[List[str]]	
	"""
	re_pred = []
	prev_len = 0
	for i in seq_len:
		seq_labels = pred[0][prev_len:i+prev_len]
		re_pred.append(list(seq_labels))
		prev_len += i
	return re_pred







