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

pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # .../deepnlp/.
sys.path.append(pkg_path)

from parse import transition_system    # absolute import
from parse import reader
from parse.transition_system import Configuration

# language option python command line 'python pos_model.py en'
lang = "zh" if len(sys.argv)==1 else sys.argv[1] # default zh
file_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(file_path, "data", lang)
train_dir = os.path.join(file_path, "ckpt", lang)

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("parse_lang", lang, "parse option for model config")
flags.DEFINE_string("parse_data_path", data_path, "data_path")
flags.DEFINE_string("parse_train_dir", train_dir, "Training directory.")
flags.DEFINE_string("parse_scope_name", "parse_var_scope", "Variable scope of pos Model")

FLAGS = flags.FLAGS

class ChineseParseConfig(object):
    """ See Chen and Manning 2014.
        vocab_size, pos_size, label_size should be aligned with corpus and reader output
    """
    init_scale = 0.04
    learning_rate = 0.1
    max_grad_norm = 10
    
    embedding_dim = 128    # vector embedding dimension
    hidden_size = 256
    max_epoch = 10
    max_max_epoch = 20
    lr_decay = 1 / 1.15
    batch_size = 1       # single sample batch
    vocab_size = 30000
    pos_size = 30
    label_size = 24     # number of dependency label
    
    feature_num = 48    # setting for features
    feature_word_num = 18  # D_word
    feature_pos_num = 18   # D_pos
    feature_label_num = 12 # D_label

class LargeConfigEnglish(object):
    init_scale = 0.04
    learning_rate = 0.1
    # To Do

def get_config(lang):
    if (lang == 'zh'):
        return ChineseParseConfig()  
    elif (lang == 'en'):
        return LargeConfigEnglish()
    # other lang options
    else :
        return None

def data_type():
    return tf.float32

class NNParser(object):
    
    def __init__(self, config):
        '''Define ANN model for Dependency Parsing Task
        '''
        feature_num = config.feature_num
        feature_word_num = config.feature_word_num
        feature_pos_num = config.feature_pos_num
        feature_label_num = config.feature_label_num
        
        self._target_num = 2 * config.label_size + 1   # left(l), right(l), shift
        embedding_dim = config.embedding_dim
        hidden_size = config.hidden_size
        self._batch_size = config.batch_size

        self.X = tf.placeholder(tf.int32, [None, feature_num])     # id of embedding
        self.Y = tf.placeholder(data_type(), [None, self._target_num])   # float
        
        self._input_w = self.X[:, 0: feature_word_num]                                    # word_id
        self._input_p = self.X[:, feature_word_num: (feature_word_num + feature_pos_num)] # pos_tag_id
        self._input_l = self.X[:, (feature_word_num + feature_pos_num): feature_num]      # arc_label_id
        
        # ANN model input layer
        with tf.device("/cpu:0"):
            embedding_w = tf.get_variable("embedding_w", [config.vocab_size, embedding_dim], dtype=data_type())
            inputs_w = tf.nn.embedding_lookup(embedding_w, self._input_w)
            inputs_w = tf.reshape(inputs_w, [self._batch_size, -1], name="inputs_w")

            embedding_p = tf.get_variable("embedding_p", [config.pos_size, embedding_dim], dtype=data_type())
            inputs_p = tf.nn.embedding_lookup(embedding_p, self._input_p)
            inputs_p = tf.reshape(inputs_p, [self._batch_size, -1], name="inputs_p")
            
            embedding_l = tf.get_variable("embedding_l", [config.label_size, embedding_dim], dtype=data_type())
            inputs_l = tf.nn.embedding_lookup(embedding_l, self._input_l)
            inputs_l = tf.reshape(inputs_l, [self._batch_size, -1], name="inputs_l")
            
        # ANN model hidden layer
        weight_w = self.weight_variable([feature_word_num * embedding_dim, hidden_size])  # [embed_dim * feature_word_num, hidden_size]
        weight_p = self.weight_variable([feature_pos_num * embedding_dim, hidden_size])
        weight_l = self.weight_variable([feature_label_num * embedding_dim, hidden_size])
        bias = self.weight_variable([hidden_size])
        
        act = tf.nn.relu  # activation function, e.g. relu, cubic
        h = act(tf.matmul(inputs_w, weight_w) + tf.matmul(inputs_p, weight_p) + tf.matmul(inputs_l, weight_l) + bias)
        
        # ANN model output layer
        weight_o = self.weight_variable([hidden_size, self._target_num]) 
        self._logit = tf.nn.softmax(tf.matmul(h, weight_o))
        self._predict_label = tf.cast(tf.argmax(self._logit, 1), tf.int32)
        self._correct_predict_num = tf.equal(tf.cast(tf.argmax(self._logit, 1), tf.int32), tf.cast(tf.argmax(self.Y, 1), tf.int32))
        self._loss = -tf.reduce_mean(self.Y * tf.log(self._logit))
        
        # Set Optimizer and learning rate
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss, tvars), config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

        self._new_lr = tf.placeholder(data_type(), shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)
        self.saver = tf.train.Saver(tf.global_variables())

    def weight_variable(self, shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def train_op(self):
        return self._train_op

    @property
    def loss(self):
        return self._loss

    @property
    def logit(self):
        return self._logit

    @property
    def predict_label(self):
        return self._predict_label

    @property
    def correct_predict_num(self):
        return self._correct_predict_num

    @property
    def lr(self):
        return self._lr

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def target_num(self):
        return self.target_num

def _crf_layer(crf_input, batch_size):
    # To Do, Need the CRF_input a sequence, not just a token
    # Linear CRF Layer
    x_crf_input = tf.reshape(output, [batch_size, num_steps, num_features])
    crf_weights = tf.get_variable("crf_weights", [num_features, target_num], dtype=data_type())
    matricized_crf_input = tf.reshape(x_crf_input, [-1, num_features])
    matricized_unary_scores = tf.matmul(matricized_crf_input, crf_weights)
    unary_scores = tf.reshape(matricized_unary_scores, [batch_size, num_steps, target_num])

    # log-likelihood
    sequence_lengths = tf.constant(np.full(batch_size, num_steps - 1, dtype=np.int32))  # shape: [batch_size], value: [T-1, T-1,...]
    log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(unary_scores, 
      targets, sequence_lengths)

    # Add a training op to tune the parameters.
    loss = tf.reduce_mean(-log_likelihood)
    logits = unary_scores  # CRF x input, shape [batch_size, num_steps, target_num]
    cost = loss
    return cost, logits, transition_params

def run_epoch(session, model, eval_op, dataset, is_train = True):
    """ Check if training process or development process
    """
    costs = 0.0
    iters = 0
    UAS_correct_num = 0
    UAS_total_num = 0
    LAS_correct_num = 0
    LAS_total_num = 0
    for step, (x, y) in enumerate(dataset):
        # x shape [batch_size, feature_size], y shape: [batch_size]
        if (is_train):
            fetches = [model.loss, model.predict_label, model.correct_predict_num, eval_op]
        else:
            fetches = [model.loss, model.predict_label, model.correct_predict_num, tf.no_op()]
        # Feeding Dict
        feed_dict = {}
        feed_dict[model.X] = x
        feed_dict[model.Y] = y
        loss, predict_label, correct_predict_num, _ = session.run(fetches, feed_dict)
        costs += loss
        iters += 1
        # Update UAS and LAS score
        LAS_correct_num += np.sum(correct_predict_num)
        LAS_total_num += np.sum(model.batch_size)
        if step % 10000 == 10:
            print("step %.3f perplexity: %.3f examples Accuracy %.3f " % (step * 1.0, np.exp(costs / iters), LAS_correct_num*1.0/LAS_total_num))
        # Save model checkpoint
        if step % 10000 == 10:
            checkpoint_path = os.path.join(FLAGS.parse_train_dir, "parser.ckpt")
            model.saver.save(session, checkpoint_path)
            print("Model Saved... at time step " + str(step))
    return np.exp(costs / iters)

def predict(session, model, sent):
    """ Generate a greedy decoding parsing of a sent object with list of [word, tag]
    """
    config = Configuration(sent)
    while not config.is_terminal():
        features = transition_system.get_features(config) # 1-D list
        # generate greedy prediction of the next_arc_id
        fetches = [model.loss, model.logit]   # fetch out prediction logit
        feed_dict = {}
        feed_dict[model.X] = features
        feed_dict[model.Y] = [0] * model.target_num    # dummy input y: 1D list of shape [, target_num]
        _, logit = session.run(fetches, feed_dict)     # not running eval_op, just for prediction
        
        pred_next_arc_id = int(np.argmax(logits))      # prediction of next arc_idx of 1 of (2*Nl +1)
        pred_next_arc = []
        config.step(pred_next_arc)                     # Configuration Take One Step 
        
        print ("next arc idx is %d and next arc is %s" % (pred_next_arc_id, pred_next_arc))
        
        next_arc = get_next_arc(config, tree)  # number
        next_arc_id = arc_labels[next_arc]

def main(_):
    raw_data = reader.load_data(FLAGS.parse_data_path)
    train_sents, train_trees, dev_sents, dev_trees, vocab_dict, pos_dict, label_dict, feature_tpl = raw_data # items in ids
    config = get_config(FLAGS.parse_lang)

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
    
    # train dataset should be generated only once and called by run_epoch function
    # max_tree_num = 8000 # for memory saving 
    for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)
        print("Epoch: %d Learning rate: %.4f" % (i + 1, session.run(m.lr)))
        # Training: Check training set perplexity
        instance_path = os.path.join(FLAGS.parse_data_path, "instance")  # saved example instance path
        train_X, train_Y = transition_system.generate_examples(train_sents, train_trees, label_dict, feature_tpl, instance_path, is_train = True)
        train_iter = transition_system.iter_examples(train_X, train_Y, m.batch_size)
        train_perplexity = run_epoch(session, m, m.train_op, train_iter)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        # Dev: Check development set perplexity
        dev_X, dev_Y = transition_system.generate_examples(dev_sents, dev_trees, label_dict, feature_tpl, instance_path, is_train = False)
        dev_iter = transition_system.iter_examples(dev_X, dev_Y, m.batch_size)
        dev_perplexity = run_epoch(session, m, tf.no_op(), dev_iter, is_train = False)
        print("Epoch: %d Dev Perplexity: %.3f" % (i + 1, dev_perplexity))

if __name__ == "__main__":
  tf.app.run()
