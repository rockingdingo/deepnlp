#!/usr/bin/python
# -*- coding:utf-8 -*-
""" 
NER tagger for building a BI-LSTM based NER tagging model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals # compatible with python3 unicode coding

import time
import numpy as np
import tensorflow as tf
import sys,os

pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # .../deepnlp/
sys.path.append(pkg_path)
from ner import reader # absolute import

# language option python command line: python ner_model_bilstm.py en
lang = "zh" if len(sys.argv)==1 else sys.argv[1] # default zh
file_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(file_path, "data", lang)  # path to find corpus vocab file
train_dir = os.path.join(file_path, "ckpt", lang)  # path to find model saved checkpoint file

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("ner_lang", lang, "ner language option for model config")
flags.DEFINE_string("ner_data_path", data_path, "data_path")
flags.DEFINE_string("ner_train_dir", train_dir, "Training directory.")
flags.DEFINE_string("ner_scope_name", "ner_var_scope", "Define NER Tagging Variable Scope Name")

FLAGS = flags.FLAGS

def data_type():
  return tf.float32

class NERTagger(object):
  """The NER Tagger Model."""

  def __init__(self, is_training, config):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.is_training = is_training
    self.crf_layer = config.crf_layer   # if the model has the final CRF decoding layer
    size = config.hidden_size
    vocab_size = config.vocab_size
    
    # Define input and target tensors
    self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])
    
    with tf.device("/cpu:0"):
      embedding = tf.get_variable("embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, self._input_data)
    
    # BiLSTM CRF model
    self._cost, self._logits, self._transition_params = _bilstm_crf_model(inputs, self._targets, config)

    # Gradients and SGD update operation for training the model.
    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars), config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())
    
    self._new_lr = tf.placeholder(data_type(), shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)
    self.saver = tf.train.Saver(tf.global_variables())
  
  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})
  
  @property
  def input_data(self):
    return self._input_data
  
  @property
  def targets(self):
    return self._targets
  
  @property
  def cost(self):
    return self._cost
  
  @property
  def logits(self):
    return self._logits

  @property
  def transition_params(self):   # transition params for CRF layer
    return self._transition_params
  
  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op
  
  @property
  def accuracy(self):
    return self._accuracy

# NER Model Configuration, Set Target Num, and input vocab_Size
class LargeConfigChinese(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 30
  hidden_size = 128
  max_epoch = 15
  max_max_epoch = 20
  keep_prob = 1.00
  lr_decay = 1 / 1.15
  batch_size = 1 # single sample batch
  vocab_size = 60000
  target_num = 8 # 7 NER Tags: nt, n, p, o, q (special), nz(entity_name), nbz(brand)
  bi_direction = True
  crf_layer = True

class LargeConfigEnglish(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 30
  hidden_size = 128
  max_epoch = 15
  max_max_epoch = 20
  keep_prob = 1.00
  lr_decay = 1 / 1.15
  batch_size = 1 # single sample batch
  vocab_size = 52000
  target_num = 15  # NER Tag 17, n, nf, nc, ne, (name, start, continue, end) n, p, o, q (special), nz entity_name, nbz
  bi_direction = True
  crf_layer = True

def get_config(lang):
  if (lang == 'zh'):
    return LargeConfigChinese()  
  elif (lang == 'en'):
    return LargeConfigEnglish()
  # other lang options
  
  else :
    return None

def _lstm_model(inputs, targets, config):
    '''
    @Use BasicLSTMCell and MultiRNNCell class to build LSTM model, 
    @return logits, cost and others
    '''
    batch_size = config.batch_size
    num_steps = config.num_steps
    num_layers = config.num_layers
    size = config.hidden_size
    vocab_size = config.vocab_size
    target_num = config.target_num # target output number    
    
    # Multiple LSTM Cells
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(size) for _ in range(num_layers)])
    initial_state = cell.zero_state(batch_size, data_type())
    
    outputs = [] # outputs shape: list of tensor with shape [batch_size, size], length: time_step
    state = initial_state
    with tf.variable_scope("ner_lstm"):
        for time_step in range(num_steps):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            (cell_output, state) = cell(inputs[:, time_step, :], state) # inputs[batch_size, time_step, hidden_size]
            outputs.append(cell_output)
    
    output = tf.reshape(tf.concat(outputs, 1), [-1, size]) # output shape [time_step, size]
    softmax_w = tf.get_variable("softmax_w", [size, target_num], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [target_num], dtype=data_type())
    logits = tf.matmul(output, softmax_w) + softmax_b
    
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(targets, [-1]), logits = logits)
    cost = tf.reduce_sum(loss)/batch_size # loss [time_step]
    
    # adding extra statistics to monitor
    correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), tf.reshape(targets, [-1]))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))    
    return cost, logits, state, initial_state

def _bilstm_model(inputs, targets, config):
    '''
    @Use BasicLSTMCell, MultiRNNCell method to build LSTM model, 
    @return logits, cost and others
    '''
    batch_size = config.batch_size
    num_steps = config.num_steps
    num_layers = config.num_layers
    size = config.hidden_size
    vocab_size = config.vocab_size
    target_num = config.target_num # target output number    

    cell_fw = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(size) for _ in range(num_layers)])
    cell_bw = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(size) for _ in range(num_layers)])

    initial_state_fw = cell_fw.zero_state(batch_size, data_type())
    initial_state_bw = cell_bw.zero_state(batch_size, data_type())
    
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    inputs_list = [tf.squeeze(s, axis = 1) for s in tf.split(value = inputs, num_or_size_splits = num_steps, axis = 1)]
    
    with tf.variable_scope("ner_bilstm"):
        outputs, state_fw, state_bw = tf.nn.static_bidirectional_rnn(
            cell_fw, cell_bw, inputs_list, initial_state_fw = initial_state_fw, 
            initial_state_bw = initial_state_bw)
    
    # outputs is a length T list of output vectors, which is [batch_size, 2 * hidden_size]
    # [time][batch][cell_fw.output_size + cell_bw.output_size]
    
    output = tf.reshape(tf.concat(outputs, 1), [-1, size * 2])
    # output has size: [T, size * 2]
    
    softmax_w = tf.get_variable("softmax_w", [size * 2, target_num], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [target_num], dtype=data_type())
    logits = tf.matmul(output, softmax_w) + softmax_b
    
    correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), tf.reshape(targets, [-1]))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(targets, [-1]), logits = logits)
    cost = tf.reduce_sum(loss)/batch_size # loss [time_step]
    return cost, logits

def _bilstm_crf_model(inputs, targets, config):
    '''
    @Use BasicLSTMCell, MultiRNNCell method to build LSTM model 
    @Use CRF layer to calculate log likelihood and viterbi decoder to caculate the optimal sequence
    @return logits, cost and others
    '''
    batch_size = config.batch_size
    num_steps = config.num_steps
    num_layers = config.num_layers
    size = config.hidden_size
    vocab_size = config.vocab_size
    target_num = config.target_num # target output number    
    num_features = 2 * size

    # Bi-LSTM NN layer
    cell_fw = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(size) for _ in range(num_layers)])
    cell_bw = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(size) for _ in range(num_layers)])
    
    initial_state_fw = cell_fw.zero_state(batch_size, data_type())
    initial_state_bw = cell_bw.zero_state(batch_size, data_type())
    
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    inputs_list = [tf.squeeze(s, axis = 1) for s in tf.split(value = inputs, num_or_size_splits = num_steps, axis = 1)]
    
    with tf.variable_scope("pos_bilstm_crf"):
        outputs, state_fw, state_bw = tf.nn.static_bidirectional_rnn(
            cell_fw, cell_bw, inputs_list, initial_state_fw = initial_state_fw, 
            initial_state_bw = initial_state_bw)
    
    # outputs is a length T list of output vectors, which is [batch_size, 2 * hidden_size]
    # [time][batch][cell_fw.output_size + cell_bw.output_size]
    
    output = tf.reshape(tf.concat(outputs, 1), [-1, num_features])
    # output has size: batch_size, [T, size * 2]
    print ("LSTM NN layer output size:")
    print (output.get_shape())

    # Linear-Chain CRF Layer
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

def run_epoch(session, model, word_data, tag_data, eval_op, verbose=False):
  """Runs the model on the given data."""
  epoch_size = ((len(word_data) // model.batch_size) - 1) // model.num_steps
  
  start_time = time.time()
  costs = 0.0
  iters = 0
  correct_labels = 0   #prediction accuracy
  total_labels = 0

  for step, (x, y) in enumerate(reader.iterator(word_data, tag_data, model.batch_size, model.num_steps)):
    feed_dict = {}
    feed_dict[model.input_data] = x
    feed_dict[model.targets] = y
    fetches = []

    if (model.crf_layer):   # model has the CRF decoding layer
      fetches = [model.cost, model.logits, model.transition_params, eval_op]
      cost, logits, transition_params, _ = session.run(fetches, feed_dict)
      # iterate over batches [batch_size, num_steps, target_num], [batch_size, target_num]
      for unary_score_, y_ in zip(logits, y):    #  unary_score_  :[num_steps, target_num], y_: [num_steps]
        viterbi_prediction = tf.contrib.crf.viterbi_decode(unary_score_, transition_params)    
        # viterbi_prediction: tuple (list[id], value)
        # y_: tuple
        correct_labels += np.sum(np.equal(viterbi_prediction[0], y_))   # compare prediction sequence with golden sequence
        total_labels += len(y_)
        #print ("step %d:" % step)
        #print ("correct_labels %d" % correct_labels)
        #print ("viterbi_prediction")
        #print (viterbi_prediction)
    else:
      fetches = [model.cost, model.logits, eval_op]
      cost, logits, _ = session.run(fetches, feed_dict)

    costs += cost
    iters += model.num_steps
    
    if verbose and step % (epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters * model.batch_size / (time.time() - start_time)))
    
    # Accuracy
    if verbose and step % (epoch_size // 10) == 20:
      accuracy = 100.0 * correct_labels / float(total_labels)
      print("Accuracy: %.2f%%" % accuracy)

    # Save Model to CheckPoint when is_training is True
    if model.is_training:
      if step % (epoch_size // 10) == 10:
        checkpoint_path = os.path.join(FLAGS.ner_train_dir, "ner_bilstm_crf.ckpt")
        model.saver.save(session, checkpoint_path)
        print("Model Saved... at time step " + str(step))
  
  return np.exp(costs / iters)

def main(_):
  if not FLAGS.ner_data_path:
    raise ValueError("No data files found in 'data_path' folder")

  raw_data = reader.load_data(FLAGS.ner_data_path)
  train_word, train_tag, dev_word, dev_tag, test_word, test_tag, vocabulary = raw_data
  
  config = get_config(FLAGS.ner_lang)
  eval_config = get_config(FLAGS.ner_lang)
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.variable_scope(FLAGS.ner_scope_name, reuse=None, initializer=initializer):
      m = NERTagger(is_training=True, config=config)
    with tf.variable_scope(FLAGS.ner_scope_name, reuse=True, initializer=initializer):
      mvalid = NERTagger(is_training=False, config=config)
      mtest = NERTagger(is_training=False, config=eval_config)
    
    # CheckPoint State
    ckpt = tf.train.get_checkpoint_state(FLAGS.ner_train_dir)
    if ckpt:
      print("Loading model parameters from %s" % ckpt.model_checkpoint_path)
      m.saver.restore(session, tf.train.latest_checkpoint(FLAGS.ner_train_dir))
    else:
      print("Created model with fresh parameters.")
      session.run(tf.global_variables_initializer())
    
    for i in range(config.max_max_epoch):
      lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
      m.assign_lr(session, config.learning_rate * lr_decay)
        
      print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
      train_perplexity = run_epoch(session, m, train_word, train_tag, m.train_op,
                                   verbose=True)
      print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
      valid_perplexity = run_epoch(session, mvalid, dev_word, dev_tag, tf.no_op())
      print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

    test_perplexity = run_epoch(session, mtest, test_word,test_tag, tf.no_op())
    print("Test Perplexity: %.3f" % test_perplexity)

if __name__ == "__main__":
  tf.app.run()
