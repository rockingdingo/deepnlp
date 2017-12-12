#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Neural Network Dependency Parsing Model
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals # compatible with python3 unicode coding

import time
import numpy as np
import tensorflow as tf
import sys,os

import parse_model
import transition_system
import reader
from reader import Sentence
from parse_model import FLAGS
from parse_model import NNParser
from parse_model import Configuration

UNKNOWN = "*"

def predict(session, model, sent, rev_arc_labels):
    """ Generate a greedy decoding parsing of a sent object with list of [word, tag]
        Return: parse tree
    """
    target_num = len(rev_arc_labels)
    print ("target number is %d" % target_num)

    config = Configuration(sent)
    while not config.is_terminal():
        features = transition_system.get_features(config) # 1-D list
        X = np.array([features])                          # convert 1-D list to ndarray size [1, dim]
        Y = np.array([[0] * target_num])

        # generate greedy prediction of the next_arc_id
        fetches = [model.loss, model.logit]   # fetch out prediction logit
        feed_dict = {}
        feed_dict[model.X] = X
        feed_dict[model.Y] = Y   # dummy input y: 1D list of shape [, target_num]
        _, logit = session.run(fetches, feed_dict)     # not running eval_op, just for prediction

        pred_next_arc_id = int(np.argmax(logit))           # prediction of next arc_idx of 1 of (2*Nl +1)
        pred_next_arc = rev_arc_labels[pred_next_arc_id]    # 5-> L(2)   idx -> L(label_id)
        config.step(pred_next_arc)                     # Configuration Take One Step 
        print ("next arc idx is %d and next arc is %s" % (pred_next_arc_id, pred_next_arc))

    # To Do Return tree
    parse_dep_tree = config.tree
    print (parse_dep_tree)
    return parse_dep_tree

def main():
    # initialize model
    config = parse_model.get_config(FLAGS.parse_lang)
    with tf.Session() as session:
        with tf.variable_scope(FLAGS.parse_scope_name):
            m = NNParser(config=config)

    # CheckPoint State
    if not os.path.exists(FLAGS.parse_train_dir):
        os.makedirs(FLAGS.parse_train_dir)

    ckpt = tf.train.get_checkpoint_state(FLAGS.parse_train_dir)
    if ckpt:
        print("Loading model parameters from %s" % ckpt.model_checkpoint_path)
        m.saver.restore(session, tf.train.latest_checkpoint(FLAGS.parse_train_dir))
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())

    # read vocab
    vocab_dict_path = os.path.join(FLAGS.parse_data_path, "vocab_dict")
    pos_dict_path = os.path.join(FLAGS.parse_data_path, "pos_dict")
    label_dict_path = os.path.join(FLAGS.parse_data_path, "label_dict")

    vocab_dict = reader._read_vocab(vocab_dict_path)
    pos_dict = reader._read_vocab(pos_dict_path)
    label_dict = reader._read_vocab(label_dict_path)

    rev_vocab_dict = reader._reverse_map(vocab_dict)
    rev_pos_dict = reader._reverse_map(pos_dict)
    rev_label_dict = reader._reverse_map(label_dict)

    print ("label_dict dict size %d" % len(label_dict))

    # get rev_label_dict
    arc_labels = transition_system.generate_arcs(label_dict)   # K:L(3), V:1   L(label_id), idx
    rev_arc_labels = reader._reverse_map(arc_labels)           # K:1,    V:L(3)
    print ("rev_arc_labels dict size %d" % len(rev_arc_labels))

    # Zh examples
    words = ['新华社', '香港', '十二月', '一日', '电']
    tags = ['NN', 'NR', 'NT', 'NT', 'NN']

    sent = Sentence()    # sentence 是保存的 (word,tag)的id
    for w, t in zip(words, tags):
        w_id = vocab_dict[w] if w in vocab_dict.keys() else vocab_dict[UNKNOWN]
        t_id = pos_dict[t] if t in pos_dict.keys() else pos_dict[UNKNOWN]
        sent.add(w_id, t_id)
    print (sent)

    # make prediction
    tree = predict(session, m, sent, rev_arc_labels)
    
    # convert tree of label_id to tree of label name
    num_token = tree.count()
    print (num_token)
    print ("ID" + "\t" + "LABEL")
    for i in range(1, num_token +1):        # print result from the 1st node, excluding ROOT node
        #cur_head_id = tree.get_head(i)      # index of head word of the ith word, e.g. 5th word,in the sentence word list
        cur_label_id = tree.get_label(i)    # index of the label in the label_dict
        
        # To Do
        #cur_head_vocab_id = sent.get_word(cur_head_id)
        #cur_head_name = rev_vocab_dict[cur_head_vocab_id] if cur_head_vocab_id in rev_vocab_dict.keys() else ""
        cur_label_name = rev_label_dict[cur_label_id] if cur_label_id in rev_label_dict.keys() else ""
        out = (str(i) + "\t" + cur_label_name)
        print (out)

if __name__ == "__main__":
    main()
