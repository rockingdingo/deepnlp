#!/usr/bin/python
# -*- coding:utf-8 -*-
""" 
pos tagger for building a LSTM based pos tagging model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals # compatible with python3 unicode coding

import time
import numpy as np
import tensorflow as tf
from tensorflow.python.tools import freeze_graph  # for future C++ reference
import sys,os

pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # .../deepnlp/
sys.path.append(pkg_path)

from pos import reader # absolute import
from model_util import get_model_var_scope
from model_util import get_config, load_config
from model_util import _pos_variables_namescope
from model_util import _pos_scope_name

# language option python command line 'python pos_model.py en'
lang = "zh" if len(sys.argv)==1 else sys.argv[1] # default zh
file_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(file_path, "data", lang)
train_dir = os.path.join(file_path, "ckpt", lang)
model_dir = os.path.join(file_path, "models", lang)   # save the graph
modle_config_path = os.path.join(file_path, "data", "models.conf")

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("pos_lang", lang, "pos language option for model config")
flags.DEFINE_string("pos_data_path", data_path, "data_path")
flags.DEFINE_string("pos_train_dir", train_dir, "Training directory.")
flags.DEFINE_string("pos_model_dir", model_dir, "Models dir for protobuf graph file")
flags.DEFINE_string("pos_scope_name", _pos_scope_name, "Variable scope of pos Model")
flags.DEFINE_string("pos_model_config_path", modle_config_path, "Model hyper parameters configuration path")

FLAGS = flags.FLAGS

def data_type():
  return tf.float32

class POSTagger(object):
  """The pos Tagger Model."""

  def __init__(self, is_training, config):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size
    target_num = config.target_num # target output number
    
    self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps], name = "input_data") # define input placeholder names
    self._targets = tf.placeholder(tf.int32, [batch_size, num_steps], name = "targets")       # define targets placeholder names
    
    # Check if Model is Training
    self.is_training = is_training
    
    # NOTICE: TF1.2 change API to make RNNcell share the same variables under namespace
    # Create multi-layer LSTM model, Separate Layers with different variables, we need to create multiple RNNCells separately
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(size) for _ in range(config.num_layers)])
    self._initial_state = cell.zero_state(batch_size, data_type())
    
    with tf.variable_scope(_pos_variables_namescope, reuse = tf.AUTO_REUSE):
      with tf.device("/cpu:0"):
        embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())
        inputs = tf.nn.embedding_lookup(embedding, self._input_data)
    
      if is_training and config.keep_prob < 1:
        inputs = tf.nn.dropout(inputs, config.keep_prob)
      
      outputs = []
      state = self._initial_state
      with tf.variable_scope("lstm", reuse = tf.AUTO_REUSE):
        for time_step in range(num_steps):
          if time_step > 0: tf.get_variable_scope().reuse_variables()
          (cell_output, state) = cell(inputs[:, time_step, :], state)
          outputs.append(cell_output)
      
      output = tf.reshape(tf.concat(outputs, 1), [-1, size])
      softmax_w = tf.get_variable(
        "softmax_w", [size, target_num], dtype=data_type())
      softmax_b = tf.get_variable("softmax_b", [target_num], dtype=data_type())
      #logits = tf.matmul(output, softmax_w) + softmax_b
      logits = tf.add(tf.matmul(output, softmax_w), softmax_b, name= "output_node")  # rename logits to output_node for future inference
    
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(self._targets, [-1]), logits = logits)
      cost = tf.reduce_sum(loss)/batch_size # loss [time_step]

      # adding extra statistics to monitor
      correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), tf.reshape(self._targets, [-1]))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Fetch Reults in session.run()
    self._cost = cost
    self._final_state = state
    self._logits = logits
    self._correct_prediction = correct_prediction

    # Set Optimizer and learning rate
    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    self._new_lr = tf.placeholder(
        data_type(), shape=[], name="new_learning_rate")
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
  def initial_state(self):
    return self._initial_state
  
  @property
  def cost(self):
    return self._cost
  
  @property
  def final_state(self):
    return self._final_state
  
  @property
  def logits(self):
    return self._logits

  @property
  def correct_prediction(self):
    return self._correct_prediction
    
  @property
  def lr(self):
    return self._lr
  
  @property
  def train_op(self):
    return self._train_op

def run_epoch(session, model, word_data, tag_data, eval_op, verbose=False):
  """Runs the model on the given data."""
  epoch_size = ((len(word_data) // model.batch_size) - 1) // model.num_steps
  start_time = time.time()
  costs = 0.0
  iters = 0
  correct_labels = 0
  total_labels = 0

  state = session.run(model.initial_state)
  for step, (x, y) in enumerate(reader.iterator(word_data, tag_data, model.batch_size,
                                                    model.num_steps)):
    fetches = [model.cost, model.final_state, model.correct_prediction,eval_op]
    feed_dict = {}
    feed_dict[model.input_data] = x
    feed_dict[model.targets] = y
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h
    cost, state, correct_prediction, _ = session.run(fetches, feed_dict)
    costs += cost
    iters += model.num_steps

    correct_labels += np.sum(correct_prediction)
    total_labels += len(correct_prediction)
    #print ("correct_labels: %d" % correct_labels)
    #print ("total_labels: %d" % total_labels)

    if verbose and step % (epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters * model.batch_size / (time.time() - start_time)))
    
    if verbose and step % (epoch_size // 10) == 10:
      accuracy = 100.0 * correct_labels / float(total_labels)
      print("Cum Accuracy: %.2f%%" % accuracy)
      
    # Save Model to CheckPoint when is_training is True
    if model.is_training:
      if step % (epoch_size // 10) == 10:
        checkpoint_path = os.path.join(FLAGS.pos_train_dir, "pos.ckpt")
        model.saver.save(session, checkpoint_path)
        print("Model Saved... at time step " + str(step))
  
  return np.exp(costs / iters)

def freeze_graph():
  checkpoint_state_name = "checkpoint_state_name"
  input_graph_name = "pos_graph.pb"
  output_graph_name = "pos_graph_output.pb"
  
  input_graph_path = os.path.join(FLAGS.pos_model_dir, input_graph_name)
  input_saver_def_path = ""
  input_binary = False
  input_checkpoint_path = os.path.join(FLAGS.pos_train_dir, 'pos.ckpt.data-00000-of-00001')

  # Note that we this normally should be only "output_node"!!!
  output_node_names = "output_node" 
  restore_op_name = "save/restore_all"
  filename_tensor_name = "save/Const:0"
  output_graph_path = os.path.join(FLAGS.pos_model_dir, output_graph_name)
  clear_devices = False

  freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                            input_binary, input_checkpoint_path,
                            output_node_names, restore_op_name,
                            filename_tensor_name, output_graph_path,
                            clear_devices)

def main(_):
  if not FLAGS.pos_data_path:
    raise ValueError("No data files found in 'data_path' folder")
  # Load Data
  raw_data = reader.load_data(FLAGS.pos_data_path)
  train_word, train_tag, dev_word, dev_tag, test_word, test_tag, vocab_size = raw_data
  
  # Load Config
  config_dict = load_config(FLAGS.pos_model_config_path)
  config = get_config(config_dict, FLAGS.pos_lang)
  eval_config = get_config(config_dict, FLAGS.pos_lang)
  eval_config.batch_size = 1
  eval_config.num_steps = 1
  
  # Define Variable Scope
  model_var_scope = get_model_var_scope(FLAGS.pos_scope_name, FLAGS.pos_lang)

  with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.variable_scope(model_var_scope, reuse=False, initializer=initializer):
      m = POSTagger(is_training=True, config=config)
    with tf.variable_scope(model_var_scope, reuse=True, initializer=initializer):
      mvalid = POSTagger(is_training=False, config=config)
      mtest = POSTagger(is_training=False, config=eval_config)
    
    # CheckPoint State
    ckpt = tf.train.get_checkpoint_state(FLAGS.pos_train_dir)
    if ckpt:
      print("Loading model parameters from %s" % ckpt.model_checkpoint_path)
      m.saver.restore(session, tf.train.latest_checkpoint(FLAGS.pos_train_dir))
    else:
      print("Created model with fresh parameters.")
      session.run(tf.global_variables_initializer())
    
    # write the graph out for further use e.g. C++ API call
    tf.train.write_graph(session.graph_def, './models/', 'pos_graph.pbtxt', as_text=True)   # output is text
    
    for i in range(config.max_max_epoch):
      lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
      m.assign_lr(session, config.learning_rate * lr_decay)

      print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
      train_perplexity = run_epoch(session, m, train_word, train_tag, m.train_op,
                                   verbose=True)
      print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
      valid_perplexity = run_epoch(session, mvalid, dev_word, dev_tag, tf.no_op())
      print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
    
    test_perplexity = run_epoch(session, mtest, test_word, test_tag, tf.no_op())
    print("Test Perplexity: %.3f" % test_perplexity)

if __name__ == "__main__":
  tf.app.run()
