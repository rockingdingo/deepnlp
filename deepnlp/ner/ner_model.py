#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
NER tagger for building a LSTM based NER tagging model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals # compatible with python3 unicode coding

import time
import numpy as np
import tensorflow as tf
import sys, os

pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # .../deepnlp/
sys.path.append(pkg_path)
from ner import reader # explicit relative import

# language option python command line 'python ner_model.py zh'
lang = "zh" if len(sys.argv)==1 else sys.argv[1] # default zh
file_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(file_path, "data", lang)
train_dir = os.path.join(file_path, "ckpt", lang)

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("ner_lang", lang, "ner language option for model config")
flags.DEFINE_string("ner_data_path", data_path, "data_path")
flags.DEFINE_string("ner_train_dir", train_dir, "Training directory.")
flags.DEFINE_string("ner_scope_name", "ner_var_scope", "Variable scope of NER Model")

FLAGS = flags.FLAGS

def data_type():
  return tf.float32

class NERTagger(object):
  """The NER Tagger Model."""

  def __init__(self, is_training, config):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size
    target_num = config.target_num # target output number
    
    self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

    # Check if Model is Training
    self.is_training = is_training
    
    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
    if is_training and config.keep_prob < 1:
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

    self._initial_state = cell.zero_state(batch_size, data_type())

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, self._input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    outputs = []
    state = self._initial_state
    with tf.variable_scope("ner_lstm"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

    output = tf.reshape(tf.concat(1, outputs), [-1, size])
    softmax_w = tf.get_variable(
        "softmax_w", [size, target_num], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [target_num], dtype=data_type())
    logits = tf.matmul(output, softmax_w) + softmax_b
    loss = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(self._targets, [-1])],
        [tf.ones([batch_size * num_steps], dtype=data_type())])
    
    # Fetch Reults in session.run()
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state
    self._logits = logits
    
    #if not is_training:
    #  return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    self._new_lr = tf.placeholder(
        data_type(), shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)
    self.saver = tf.train.Saver(tf.all_variables())


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
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

# NER Model Configuration, Set Target Num, and input vocab_Size
class LargeConfigChinese(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 30
  hidden_size = 128
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.95
  lr_decay = 1 / 1.15
  batch_size = 1 # single sample batch
  vocab_size = 52000
  target_num = 19  # NER Tag 17, n, nf, nc, ne, (name, start, continue, end) n, p, o, q (special), nz entity_name, nbz

class LargeConfigEnglish(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 30
  hidden_size = 128
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.95
  lr_decay = 1 / 1.15
  batch_size = 1 # single sample batch
  vocab_size = 52000
  target_num = 15  # NER Tag 17, n, nf, nc, ne, (name, start, continue, end) n, p, o, q (special), nz entity_name, nbz

def get_config(lang):
  if (lang == 'zh'):
    return LargeConfigChinese()  
  elif (lang == 'en'):
    return LargeConfigEnglish()
  # other lang options
  
  else :
    return None

def run_epoch(session, model, word_data, tag_data, eval_op, verbose=False):
  """Runs the model on the given data."""
  epoch_size = ((len(word_data) // model.batch_size) - 1) // model.num_steps
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)
  for step, (x, y) in enumerate(reader.iterator(word_data, tag_data, model.batch_size,
                                                    model.num_steps)):
    fetches = [model.cost, model.final_state, eval_op]
    feed_dict = {}
    feed_dict[model.input_data] = x
    feed_dict[model.targets] = y
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h
    cost, state, _ = session.run(fetches, feed_dict)
    costs += cost
    iters += model.num_steps

    if verbose and step % (epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters * model.batch_size / (time.time() - start_time)))
    
    # Save Model to CheckPoint when is_training is True
    if model.is_training:
      if step % (epoch_size // 10) == 10:
        checkpoint_path = os.path.join(FLAGS.ner_train_dir, "ner.ckpt")
        model.saver.save(session, checkpoint_path)
        print("Model Saved... at time step " + str(step))

  return np.exp(costs / iters)


def main(_):
  if not FLAGS.ner_data_path:
    raise ValueError("No data files found in 'data_path' folder")

  raw_data = reader.load_data(FLAGS.ner_data_path)
  # train_data, valid_data, test_data, _ = raw_data
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
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
      print("Loading model parameters from %s" % ckpt.model_checkpoint_path)
      m.saver.restore(session, ckpt.model_checkpoint_path)
    else:
      print("Created model with fresh parameters.")
      session.run(tf.initialize_all_variables())

    # tf.initialize_all_variables().run()

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
