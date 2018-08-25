#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
#from text_cnn_rnn import TextCNNRNN
#from text_rnn_cnn import TextRNNCNN
#from text_rnncnn import TextRNNandCNN
#from text_rnn import TextRNN
from tensorflow.contrib import learn
import csv
import json
import pdb
# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("model_type", "cnn", "model type cnn or cnnrnn or rnn, rnncnn, rnnandcnn")
tf.flags.DEFINE_string("test_data_file", "./data/cnews.test.seg", "test data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")
tf.flags.DEFINE_boolean("topk_eval", True, "get topk result or result that higher than a score(0.5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

def get_real_len(x_text, max_len):
    real_len = []
    for item in x_text:
        tmp_list = item.split(" ")
        seq_len = len(tmp_list)
        if seq_len > max_len:
            seq_len = max_len
        real_len.append(seq_len)
    return real_len

def load_train_params(train_dir):
    sorted_label = json.loads(open(train_dir + '/sorted_label.json').read())
    train_params = json.loads(open(train_dir + '/train_params.json').read())
    return sorted_label,train_params

def get_label_using_logits(predictions, topk=3):
    #get top k result according to sigmoid score
    #input: predictions: numpy array 
    top_result = []
    for pred in predictions:
        index_list = np.argsort(pred)[-topk:] # get topk index
        index_list = index_list[::-1] # sort from high score to low score
        top_result.append(index_list)#get top k pred result for every sample
    return top_result

def calculate_accuracy_topk_label(pred_inx_list, act_inx_list):
    err_cnt = 0
    assert len(pred_inx_list) == len(act_inx_list)
    total_cnt = len(pred_inx_list)
    for pred_list, act_list in zip(pred_inx_list, act_inx_list):
        for pred in pred_list:
            if pred not in act_list:
                err_cnt += 1
                break
    acc_cnt = total_cnt - err_cnt
    acc = 1.0 * acc_cnt / total_cnt
    return acc

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    train_dir = os.path.join(FLAGS.checkpoint_dir, "..", "trained_results")
    sorted_label, train_params = load_train_params(train_dir)
    x_raw, y_test = data_helpers.load_test_data(FLAGS.test_data_file, sorted_label)
    #y_test = np.argmax(y_test, axis=1)
else:
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_real_len_test = np.array(get_real_len(x_raw, train_params['max_document_length']))
x_test = np.array(list(vocab_processor.transform(x_raw)))
print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        #pdb.set_trace()
        print("Reading model parameters from %s" % checkpoint_file)
        if FLAGS.model_type == "cnnrnn" or FLAGS.model_type == "rnncnn" or FLAGS.model_type == "rnn" or FLAGS.model_type == "rnnandcnn":
            real_len = graph.get_operation_by_name("real_len").outputs[0]
        else:
            is_training = graph.get_operation_by_name("is_training").outputs[0]
        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        correct_pred_num = graph.get_operation_by_name("accuracy/correct_num").outputs[0]
        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(zip(x_test, x_real_len_test, y_test)), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        #all_predictions = []
        correct_total_num = 0
        total_pred_inx_list = []
        total_act_inx_list = []
        for batch in batches:
            x_test_batch, x_real_len_test_batch, y_test_batch = zip(*batch)
            if FLAGS.model_type == "cnn":
                batch_predictions, batch_correct = sess.run([predictions, correct_pred_num], {input_x: x_test_batch, input_y: y_test_batch, dropout_keep_prob: 1.0, is_training: False})
            else:
                batch_predictions, batch_corrct = sess.run([predictions, correct_pred_num], {input_x: x_test_batch, input_y: y_test_batch, dropout_keep_prob: 1.0, real_len: x_real_len_test_batch})
            pred_inx_list = get_label_using_logits(batch_predictions, topk=1)
            act_inx_list = get_label_using_logits(y_test_batch, topk=1)
            total_pred_inx_list += pred_inx_list
            total_act_inx_list += act_inx_list
            #all_predictions = np.concatenate([all_predictions, batch_predictions])
            correct_total_num += batch_correct

# Print accuracy if y_test is defined
if y_test is not None:
    #correct_predictions = float(sum(all_predictions == y_test))
    topk_acc = calculate_accuracy_topk_label(total_pred_inx_list, total_act_inx_list)
    correct_predictions = correct_total_num
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
    print("Topk Accuracy: {:g}".format(topk_acc))

"""
# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
"""
