import tensorflow as tf

#----------------------------------- Base Model --------------------------------------
def create_tensorflow_model(uni_vocab_size, bi_vocab_size, uni_embedding_size, bi_embedding_size, hidden_size):
	print("Creating TENSORFLOW model")
	 
	# Unigrams Inputs have (batch_size, timesteps) shape.
	uni_inputs = tf.placeholder(tf.int32, shape=[None, None])

	# Bigrams Input have (batch_size, timesteps) shape.
	bi_inputs  = tf.placeholder(tf.int32, shape=[None, None])

	# Labels have (batch_size,) shape.
	labels     = tf.placeholder(tf.int64, shape=[None, None])
	# Keep_prob is a scalar.
	keep_prob  = tf.placeholder(tf.float32, shape=[])
	# Calculate sequence lengths to mask out the paddings later on.
	seq_length = tf.count_nonzero(uni_inputs, axis=-1)
	# Create mask using sequence mask
	# pad_mask   = tf.sequence_mask(seq_length)
	pad_mask   = tf.to_float(tf.not_equal(uni_inputs, 0))


	with tf.variable_scope("uni_embeddings"):
		uni_embedding_matrix = tf.get_variable("uni_embeddings", shape=[uni_vocab_size, uni_embedding_size])
		uni_embeddings = tf.nn.embedding_lookup(uni_embedding_matrix, uni_inputs)
	
	with tf.variable_scope("bi_embeddings"):
		bi_embedding_matrix = tf.get_variable("bi_embeddings", shape=[bi_vocab_size, bi_embedding_size])
		bi_embeddings = tf.nn.embedding_lookup(bi_embedding_matrix, bi_inputs)

	with tf.variable_scope("concat_embeddings"):
		concat_embeddings = tf.concat([uni_embeddings, bi_embeddings], axis=2)

	with tf.variable_scope("rnn"):

		rnn_cell_fwd = tf.nn.rnn_cell.LSTMCell(hidden_size)
		rnn_cell_bwd = tf.nn.rnn_cell.LSTMCell(hidden_size)

		# Add dropout to the LSTM cell
		rnn_cell_fwd = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_fwd,
			input_keep_prob=keep_prob, 
			output_keep_prob=keep_prob,
			state_keep_prob=keep_prob)

		rnn_cell_bwd = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_bwd,
			input_keep_prob=keep_prob,
			output_keep_prob=keep_prob,
			state_keep_prob=keep_prob)

		outputs, _ = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fwd, rnn_cell_bwd, concat_embeddings, sequence_length=seq_length, dtype=tf.float32)

		concat_out = tf.concat(outputs, 2)


	with tf.variable_scope("dense"):
		logits = tf.layers.dense(concat_out, 4, activation=None)
		logits = tf.squeeze(logits)

	with tf.variable_scope("loss"):
		padded_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
		masked_loss = tf.boolean_mask(padded_loss, pad_mask)
		loss        = tf.reduce_mean(masked_loss)

	with tf.variable_scope("train"):
		# train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
		train_op    = tf.train.MomentumOptimizer(0.04, 0.95).minimize(loss)

	with tf.variable_scope("accuracy"):
		predictions   = tf.boolean_mask(tf.cast(tf.argmax(logits, axis=-1), tf.int64), pad_mask)
		masked_labels = tf.boolean_mask(labels, pad_mask)
		masked_acc    = tf.equal(predictions, masked_labels)
		
		acc = tf.reduce_mean(tf.cast(masked_acc, tf.float32))

	return uni_inputs, bi_inputs, labels, keep_prob, loss, train_op, acc, predictions

































#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------







#----------------------------------- Experiments Model - 1 --------------------------------------
def Exp_model(uni_vocab_size, bi_vocab_size, uni_embedding_size, bi_embedding_size, hidden_size, batch_size, n_iterations):
	print("Creating TENSORFLOW model")
	 
	# Unigrams Inputs have (batch_size, timesteps) shape.
	uni_inputs = tf.placeholder(tf.int32, shape=[None, None])

	# Bigrams Input have (batch_size, timesteps) shape.
	bi_inputs  = tf.placeholder(tf.int32, shape=[None, None])

	# Labels have (batch_size,) shape.
	labels     = tf.placeholder(tf.int64, shape=[None, None])
	# Keep_prob is a scalar.
	keep_prob  = tf.placeholder(tf.float32, shape=[])
	# Calculate sequence lengths to mask out the paddings later on.
	seq_length = tf.count_nonzero(uni_inputs, axis=-1)
	# Create mask using sequence mask
	# pad_mask   = tf.sequence_mask(seq_length)
	pad_mask   = tf.to_float(tf.not_equal(uni_inputs, 0))
	#For exponential decay of learning rate with each step
	batch = tf.Variable(0)
	#xavier initializer for trials
    #xav_init   = tf.contrib.layers.xavier_initializer(uniform=False)

	with tf.variable_scope("uni_embeddings"):
		uni_embedding_matrix = tf.get_variable("uni_embeddings", shape=[uni_vocab_size, uni_embedding_size])
		uni_embeddings = tf.nn.embedding_lookup(uni_embedding_matrix, uni_inputs)
	
	with tf.variable_scope("bi_embeddings"):
		bi_embedding_matrix = tf.get_variable("bi_embeddings", shape=[bi_vocab_size, bi_embedding_size])
		bi_embeddings = tf.nn.embedding_lookup(bi_embedding_matrix, bi_inputs)

	with tf.variable_scope("concat_embeddings"):
		concat_embeddings = tf.concat([uni_embeddings, bi_embeddings], axis=2)

	with tf.variable_scope("rnn"):

		rnn_cell_fwd = tf.nn.rnn_cell.LSTMCell(hidden_size)
		rnn_cell_bwd = tf.nn.rnn_cell.LSTMCell(hidden_size)

		# Add dropout to the LSTM cell
		rnn_cell_fwd = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_fwd,
			input_keep_prob=keep_prob, 
			output_keep_prob=keep_prob,
			state_keep_prob=keep_prob)

		rnn_cell_bwd = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_bwd,
			input_keep_prob=keep_prob,
			output_keep_prob=keep_prob,
			state_keep_prob=keep_prob)

		outputs, _ = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fwd, rnn_cell_bwd, concat_embeddings, sequence_length=seq_length, dtype=tf.float32)

		concat_out = tf.concat(outputs, 2)


	with tf.variable_scope("dense"):
		
		pre_logits = tf.layers.dense(concat_out, 16, activation=tf.nn.tanh)

		logits = tf.layers.dense(pre_logits, 4, activation=None)
		logits = tf.squeeze(logits)

	#
	with tf.variable_scope("loss"):
		padded_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits = logits)
		masked_loss = tf.boolean_mask(padded_loss, pad_mask)
		loss        = tf.reduce_mean(masked_loss)

	with tf.variable_scope("train-decay-nesterov"):
		learning_rate = tf.train.exponential_decay(
		  0.045,               # Base learning rate.
		  batch,  			   # Current index into the dataset.
		  n_iterations,		   # Decay step.
		  0.95,                # Decay rate.
		  staircase=True)

		train_op = tf.train.MomentumOptimizer(learning_rate, 0.95, 
														use_nesterov=True).minimize(loss,
		                                               	global_step=batch)


	# with tf.variable_scope("train"):
	# 	# train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
	# 	train_op    = tf.train.MomentumOptimizer(0.04, 0.95).minimize(loss)

	with tf.variable_scope("accuracy"):
		predictions   = tf.boolean_mask(tf.cast(tf.argmax(logits, axis=-1), tf.int64), pad_mask)
		masked_labels = tf.boolean_mask(labels, pad_mask)
		masked_acc    = tf.equal(predictions, masked_labels)
		
		acc = tf.reduce_mean(tf.cast(masked_acc, tf.float32))

	return uni_inputs, bi_inputs, labels, keep_prob, loss, train_op, acc, predictions












































#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------








#----------------------------------- Experiments Model - CRF - Self Attention --------------------------------------
def crf_att_tf_model(uni_vocab_size, bi_vocab_size, uni_embedding_size, bi_embedding_size, hidden_size, batch_size, n_iterations, max_length):
	print("Creating TENSORFLOW model")
	 
	# Unigrams Inputs have (batch_size, timesteps) shape.
	uni_inputs = tf.placeholder(tf.int32, shape=[None, None])

	# Bigrams Input have (batch_size, timesteps) shape.
	bi_inputs  = tf.placeholder(tf.int32, shape=[None, None])

	# Labels have (batch_size,) shape.
	labels     = tf.placeholder(tf.int32, shape=[None, None])
	# Keep_prob is a scalar.
	keep_prob  = tf.placeholder(tf.float32, shape=[])
	# Calculate sequence lengths to mask out the paddings later on.
	seq_length = tf.count_nonzero(uni_inputs, axis=-1)
	# Create mask using sequence mask
	# pad_mask   = tf.sequence_mask(seq_length)
	pad_mask   = tf.to_float(tf.not_equal(uni_inputs, 0))
	#For exponential decay of learning rate with each step
	batch = tf.Variable(0)
	#xavier initializer for trials
	xav_init   = tf.contrib.layers.xavier_initializer(uniform=False)

	with tf.variable_scope("uni_embeddings"):
		uni_embedding_matrix = tf.get_variable("uni_embeddings", shape=[uni_vocab_size, uni_embedding_size]
																						, initializer=xav_init)
		uni_embeddings = tf.nn.embedding_lookup(uni_embedding_matrix, uni_inputs)
	
	with tf.variable_scope("bi_embeddings"):
		bi_embedding_matrix = tf.get_variable("bi_embeddings", shape=[bi_vocab_size, bi_embedding_size]
																						, initializer=xav_init)
		bi_embeddings = tf.nn.embedding_lookup(bi_embedding_matrix, bi_inputs)

	with tf.variable_scope("concat_embeddings"):
		concat_embeddings = tf.concat([uni_embeddings, bi_embeddings], axis=2)

	with tf.variable_scope("rnn"):

		rnn_cell_fwd = tf.nn.rnn_cell.LSTMCell(hidden_size)
		rnn_cell_bwd = tf.nn.rnn_cell.LSTMCell(hidden_size)

		# Add dropout to the LSTM cell
		rnn_cell_fwd = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_fwd,
			input_keep_prob=keep_prob, 
			output_keep_prob=keep_prob,
			state_keep_prob=keep_prob)

		rnn_cell_bwd = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_bwd,
			input_keep_prob=keep_prob,
			output_keep_prob=keep_prob,
			state_keep_prob=keep_prob)

		outputs, _ = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fwd, rnn_cell_bwd, concat_embeddings, sequence_length=seq_length, dtype=tf.float32)
	
		concat_out = tf.concat(outputs, 2)


	with tf.variable_scope("dense"):
		hidden_size_att = concat_out.shape[2].value
		epsilon = 1e-8

		# Trainable parameters
		omega_att = tf.get_variable(shape=[hidden_size_att, 1], dtype=tf.float32,
		                        initializer=tf.truncated_normal_initializer(stddev=0.05), name='omega')

		omega_H = tf.tensordot(concat_out, omega_att, axes=1) 
		omega_H = tf.squeeze(omega_H, -1) 

		u = tf.tanh(omega_H)  

		# Softmax
		a = tf.exp(u)
		a = a * pad_mask
		a /= tf.reduce_sum(a, axis=1, keepdims=True) + epsilon 

		w_inputs = concat_out * tf.expand_dims(a, -1)

		c = tf.reduce_sum(w_inputs, axis=1)

		c_expanded = tf.expand_dims(c,axis=1)
		attention_vec = tf.tile(c_expanded, [1,max_length,1])
		self_att_concat = tf.concat([concat_out,attention_vec], 2)

	with tf.variable_scope("CRF-loss"):
		w_CRF = tf.get_variable("W", shape=[4*hidden_size, 4],
                dtype=tf.float32)

		b_CRF = tf.get_variable("b", shape=[4], dtype=tf.float32,
                initializer=tf.zeros_initializer())

		ntime_steps = tf.shape(self_att_concat)[1]
		context_rep_flat = tf.reshape(self_att_concat, [-1, 4*hidden_size])
		pred   = tf.matmul(context_rep_flat, w_CRF) + b_CRF
		scores = tf.reshape(pred, [-1, ntime_steps, 4])
		log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
															scores, labels,
															seq_length)

		loss = tf.reduce_mean(-log_likelihood)

	with tf.variable_scope("train-decay"):
		learning_rate = tf.train.exponential_decay(
		  0.045,               # Base learning rate.
		  batch,  			   # Current index into the dataset.
		  n_iterations,		   # Decay step.
		  0.95,                # Decay rate.
		  staircase=True)
		train_op = tf.train.MomentumOptimizer(learning_rate, 0.95, 
														use_nesterov=True).minimize(loss,
		                                               	global_step=batch)


	with tf.variable_scope("accuracy"):

		viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(scores, transition_params, seq_length)
		predictions   = tf.boolean_mask(viterbi_sequence, pad_mask)
		masked_labels = tf.boolean_mask(labels, pad_mask)
		masked_acc    = tf.equal(predictions, masked_labels)
		
		acc = tf.reduce_mean(tf.cast(masked_acc, tf.float32))

	return uni_inputs, bi_inputs, labels, keep_prob, loss, train_op, acc, predictions



