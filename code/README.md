# Scripts
pre.py takes in a the raw input and returns the processed input file (without spaces) and the labels

Concat.py concatenates the raw training and dev files required for the experiments

pre_dict.py takes in the processed input file and returns the required dictionaries of ngrams and labels

pre_tf.py contains the actual model and does the training

utility.py contains the utility functions

tf_model.py contains the skelatons of all the models used during the experiments

predict.py produces a text file with the predictions for the input file 

pred_utility.py contains the helper functions to execute the prediction

score provides the precision of the prediction 
