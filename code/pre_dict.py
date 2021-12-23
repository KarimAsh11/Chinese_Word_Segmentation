import pickle
import argparse
from collections import defaultdict
from utility import read

parser = argparse.ArgumentParser(description='Dataset preparation')

parser.add_argument('input',
                    help='input file')

parser.add_argument('dict_dir',
                    help='dictionary directory')

parser.add_argument('vocab_size', type=int, default=100000, nargs='?', const=1,
                    help='vocab size')


def _save_pickle(dictionary, pickle_file):
    """
    Pickle dictionary.

    params:
        dictionary (dict)
        pickle_file (str)
    """
    with open(pickle_file, 'wb') as h:
        pickle.dump(dictionary, h, protocol=pickle.HIGHEST_PROTOCOL)


def _find_ngrams(sentence, n, end_token):
    """
    returns list of all ngrams from sample.

    params:
        sentence (str)
        n (int)
        end_token (boolean)
 
    returns: list
    """
    if n>1 and end_token:   
        sentence.append("</S>")

    ngrams=zip(*[sentence[i:] for i in range(n)])

    return ["".join(ngram) for ngram in ngrams]


def _trim_ngrams(ngrams, vocab_size):
    """_trim_ngrams()
    returns list of n most frequent ngrams from dataset.

    params:
        ngrams (list)
        n (int)
 
    returns: list
    """
    d = defaultdict(int)

    for ngram in ngrams:
        d[ngram] += 1

    return [k for k in sorted(d, key=d.get, reverse=True)[:vocab_size]]


def _set_ngrams(file, n, end_token, vocab_size, trim):
    """
    returns set of ngrams in dataset.

    params:
        file (str)
        n (int)
        end_token (boolean)
        vocab_size (int)
        trim (boolean)

    returns: list
    """
    ngrams = [] #List with all ngrams

    samples = file.splitlines()
    for sample in samples:
        ngrams.extend(_find_ngrams(list(sample), n, end_token))

    if trim:
        return _trim_ngrams(ngrams,vocab_size)
    else:
        return list(set(ngrams))


def _word_to_id(ngrams):
    """
    Get word to id dictionary.

    params:
        ngrams (list)

    returns: dict
    """
    word_to_id = dict()
    word_to_id["<PAD>"] = 0 #zero is not casual!
    word_to_id["<UNK>"] = 1 #OOV are mapped as <UNK>
    word_to_id.update({w:i+len(word_to_id) for i, w in enumerate(ngrams)})
    return word_to_id


def _id_to_word(word_to_id):
    """
    Get id to word dictionary.

    params:
        word_to_id (dict)

    returns: dict
    """
    id_to_word = {v:k for k,v in word_to_id.items()}
    return id_to_word


def _label_to_id():
    """
    Get label to id dictionary.

    params:

    returns: dict
    """
    label_to_id = dict()
    label_to_id["B"] = 0 
    label_to_id["E"] = 1 
    label_to_id["I"] = 2 
    label_to_id["S"] = 3

    return label_to_id


def run(input, dict_dir, vocab_size):
    """
    Create ngram and label dictionaries.

    params:
        input (str): input file path
        dir (str): directory path to save dictionaries in
    """
    file     = read(input)

    unigrams = _set_ngrams(file, 1, False, vocab_size, False)
    bigrams  = _set_ngrams(file, 2, True, vocab_size, True)

    uni_to_id = _word_to_id(unigrams)
    bi_to_id  = _word_to_id(bigrams)
    
    id_to_uni = _id_to_word(uni_to_id)
    id_to_bi  = _id_to_word(bi_to_id)

    label_to_id = _label_to_id()
    id_to_label  = _id_to_word(label_to_id)


    _save_pickle(uni_to_id, dict_dir+"/uni_to_id.pickle")
    _save_pickle(bi_to_id, dict_dir+"/bi_to_id.pickle")

    _save_pickle(id_to_uni, dict_dir+"/id_to_uni.pickle")
    _save_pickle(id_to_bi, dict_dir+"/id_to_bi.pickle")

    _save_pickle(label_to_id, dict_dir+"/label_to_id.pickle")
    _save_pickle(id_to_label, dict_dir+"/id_to_label.pickle")


if __name__ == '__main__':
    args = parser.parse_args()
    run(**vars(args))
