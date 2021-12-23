import tensorflow as tf
import numpy as np 

from tf_model import crf_att_tf_model
from utility import load_pickle, read, encode_to_id


# #Paths for dictionaries to be used for training - gcloud and local
uni_dict_file       = "../Resources/Dicts/uni_to_id.pickle"
bi_dict_file        = "../Resources/Dicts/bi_to_id.pickle"
label_dict_file     = "../Resources/Dicts/label_to_id.pickle"

#Paths for txt files to be used for training and dev
input_train_file     = "../Data/Processed/concat_input_training.txt"
label_train_file     = "../Data/Processed/concat_labels_training.txt"
input_dev_file       = "../Data/Processed/concat_input_dev.txt"
label_dev_file       = "../Data/Processed/concat_labels_dev.txt"

#CONSTANTS
MAX_LENGTH         = 80
UNI_EMBEDDING_SIZE = 32
BI_EMBEDDING_SIZE  = 32
HIDDEN_SIZE        = 128

#Reading txt Files
input_train     = read(input_train_file)
label_train     = read(label_train_file)
input_dev       = read(input_dev_file)
label_dev       = read(label_dev_file)

uni_dict        = load_pickle(uni_dict_file)
bi_dict         = load_pickle(bi_dict_file)
label_dict      = load_pickle(label_dict_file)


UNI_VOCAB_SIZE  = len(uni_dict) #Length of unigram dictionary
BI_VOCAB_SIZE   = len(bi_dict)  #Length of bigram dictionary
CLASS_NUMBER    = len(label_dict) #Number of classes for task
NUMBER_OF_EX    = len(input_train.splitlines()) #Num of example file


uni_train_set   = encode_to_id(input_train, uni_dict, MAX_LENGTH, False)
bi_train_set    = encode_to_id(input_train, bi_dict, MAX_LENGTH, True)
lbl_train_set   = encode_to_id(label_train, label_dict, MAX_LENGTH, False)

uni_dev_set     = encode_to_id(input_dev, uni_dict, MAX_LENGTH, False)
bi_dev_set      = encode_to_id(input_dev, bi_dict, MAX_LENGTH, True)
lbl_dev_set     = encode_to_id(label_dev, label_dict, MAX_LENGTH, False)



def batch_generator(X1, X2, Y, batch_size, shuffle=False):
	if not shuffle:
		for start in range(0, len(X1), batch_size):
			end = start + batch_size
			yield X1[start:end], X2[start:end], Y[start:end]
	else:
		perm = np.random.permutation(len(X1))
		for start in range(0, len(X1), batch_size):
			end = start + batch_size
			yield X1[perm[start:end]], X2[perm[start:end]], Y[perm[start:end]]



def add_summary(writer, name, value, global_step):
	summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
	writer.add_summary(summary, global_step=global_step)



epochs           = 10 
batch_size       = 32
n_iterations     = int(np.ceil(len(uni_train_set)/batch_size))
n_dev_iterations = int(np.ceil(len(uni_dev_set)/batch_size))

uni_inputs, bi_inputs, labels, keep_prob, loss, train_op, acc, predictions = crf_att_tf_model(UNI_VOCAB_SIZE, BI_VOCAB_SIZE, UNI_EMBEDDING_SIZE, BI_EMBEDDING_SIZE, HIDDEN_SIZE, batch_size, n_iterations, MAX_LENGTH)

saver = tf.train.Saver()
with tf.Session() as sess:
	print("\nStarting training...")
	sess.run(tf.global_variables_initializer())
	sess.run(tf.initializers.local_variables())
	train_writer = tf.summary.FileWriter('logging/tensorflow_model', sess.graph)

	for epoch in range(epochs):
		print("\nEpoch", epoch + 1)
		epoch_loss, epoch_acc = 0., 0.
		mb = 0
		print("======="*10)

		for batch_x1, batch_x2, batch_y in batch_generator(uni_train_set, bi_train_set, lbl_train_set, batch_size, shuffle=True):
			mb += 1
			#NO DROP OUT HERE
			_, loss_val, acc_val = sess.run([train_op, loss, acc], 
			                                feed_dict={uni_inputs: batch_x1, bi_inputs: batch_x2, labels: batch_y, keep_prob: 1})

			epoch_loss += loss_val
			epoch_acc += acc_val

			print("Epoch:{} - {:.2f}\tTrain Loss: {:.4f}\tTrain Accuracy: {:.4f} ".format(epoch + 1, 100.*mb/n_iterations, epoch_loss/mb, epoch_acc/mb), end="\t")

		epoch_loss /= n_iterations
		epoch_acc  /= n_iterations
		add_summary(train_writer, "epoch_loss", epoch_loss, epoch)
		add_summary(train_writer, "epoch_acc", epoch_acc, epoch)
		print("\n")
		print("\nEpoch", epoch + 1)
		print("Train Loss: {:.4f}\tTrain Accuracy: {:.4f}".format(epoch_loss, epoch_acc))
		print("======="*10)
		#Save model after each epoch
		model_path = "./tmp/epoch_"+str(epoch+1)+"_model.ckpt"
		save_path  = saver.save(sess, model_path)

		# DEV - EVALUATION
		dev_loss, dev_acc = 0.0, 0.0
		for batch_x1, batch_x2, batch_y in batch_generator(uni_dev_set, bi_dev_set, lbl_dev_set, batch_size):
			loss_val, acc_val = sess.run([loss, acc], feed_dict={uni_inputs: batch_x1, bi_inputs: batch_x2, labels: batch_y, keep_prob: 1.0})
			dev_loss += loss_val
			dev_acc  += acc_val
		dev_loss /= n_dev_iterations
		dev_acc /= n_dev_iterations

		add_summary(train_writer, "epoch_val_loss", dev_loss, epoch)
		add_summary(train_writer, "epoch_val_acc", dev_acc, epoch)
		print("\nDev Loss: {:.4f}\tDev Accuracy: {:.4f}".format(dev_loss, dev_acc))
	train_writer.close()

	save_path = saver.save(sess, "./tmp/final_model.ckpt")

	#TEST EVALUATION
	print("\nEvaluating test...")
	n_test_iterations = int(np.ceil(len(uni_dev_set)/batch_size))
	test_loss, test_acc = 0.0, 0.0
	for batch_x1, batch_x2, batch_y in batch_generator(uni_dev_set, bi_dev_set, lbl_dev_set, batch_size):
		loss_val, acc_val = sess.run([loss, acc], feed_dict={uni_inputs: batch_x1, bi_inputs: batch_x2, labels: batch_y, keep_prob: 1.0})
		test_loss += loss_val
		test_acc += acc_val
	test_loss /= n_test_iterations
	test_acc /= n_test_iterations
	print("\nTest Loss: {:.4f}\tTest Accuracy: {:.4f}".format(test_loss, test_acc))
