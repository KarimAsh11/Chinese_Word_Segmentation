import tensorflow as tf

from argparse import ArgumentParser
from tf_model import crf_att_tf_model
from pred_utility import load_hp, write_pred, load_ckpt, pred_helper

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_path", help="The path of the input file")
    parser.add_argument("output_path", help="The path of the output file")
    parser.add_argument("resources_path", help="The path of the resources needed to load your model")

    return parser.parse_args()


def predict(input_path, output_path, resources_path):
    """
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the BIES format.
    
    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.
    
    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.

    :param input_path: the path of the input file to predict.
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """

    MAX_LENGTH, UNI_EMBEDDING_SIZE, BI_EMBEDDING_SIZE, HIDDEN_SIZE = load_hp()
    UNI_VOCAB_SIZE,  BI_VOCAB_SIZE, uni_train_set, bi_train_set, orig_len = pred_helper(input_path, resources_path, MAX_LENGTH)
    model_ckpt = load_ckpt(resources_path)
    print(HIDDEN_SIZE)
    print(UNI_EMBEDDING_SIZE)

    uni_inputs, bi_inputs, labels, keep_prob, loss, train_op, acc, predictions = crf_att_tf_model(UNI_VOCAB_SIZE, BI_VOCAB_SIZE, UNI_EMBEDDING_SIZE, BI_EMBEDDING_SIZE, HIDDEN_SIZE, 0, 0, MAX_LENGTH)
    saver = tf.train.Saver()
    
    
    with tf.Session() as sess:
        print("\nStarting prediction...")
        sess.run(tf.global_variables_initializer())

        saver.restore(sess, tf.train.latest_checkpoint(model_ckpt))
        predic = sess.run([predictions], feed_dict={uni_inputs: uni_train_set, bi_inputs: bi_train_set, keep_prob: 1})

    write_pred(predic, orig_len, output_path)


if __name__ == '__main__':
    args = parse_args()
    predict(args.input_path, args.output_path, args.resources_path)
