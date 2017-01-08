#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Automatic Summarization: Generating News Headline Seq2Seq Model implementation
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange
import tensorflow as tf

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # parent folder
sys.path.append(parent_dir)

#from textsum import data_utils # absolute import
#from textsum import seq2seq_model # absolute import

import data_utils
import seq2seq_model

file_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(file_path, "news")
train_dir = os.path.join(file_path, "ckpt")

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
# article length padded to 120 and summary padded to 30
buckets = [(120, 30), (200, 35), (300, 40), (400, 40), (500, 40)]

class LargeConfig(object):
    learning_rate = 1.0
    init_scale = 0.04
    learning_rate_decay_factor = 0.99
    max_gradient_norm = 5.0
    num_samples = 4096 # Sampled Softmax
    batch_size = 64
    size = 256 # Number of Node of each layer
    num_layers = 4
    vocab_size = 50000

class MediumConfig(object):
    learning_rate = 0.5
    init_scale = 0.04
    learning_rate_decay_factor = 0.99
    max_gradient_norm = 5.0
    num_samples = 2048 # Sampled Softmax
    batch_size = 64
    size = 64 # Number of Node of each layer
    num_layers = 2
    vocab_size = 10000

config = LargeConfig() # new Large Config, set to tf.app.flags
# config = MediumConfig()

tf.app.flags.DEFINE_float("learning_rate", config.learning_rate, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", config.learning_rate_decay_factor, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", config.max_gradient_norm, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("num_samples", config.num_samples, "Number of Samples for Sampled softmax")
tf.app.flags.DEFINE_integer("batch_size", config.batch_size, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", config.size, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", config.num_layers, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("vocab_size", config.vocab_size, "vocabulary size.")

tf.app.flags.DEFINE_string("data_dir", data_path, "Data directory")
tf.app.flags.DEFINE_string("train_dir", train_dir, "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 1000, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding.") # true for prediction
tf.app.flags.DEFINE_boolean("use_fp16", False, "Train using fp16 instead of fp32.")

# define namespace for this model only
tf.app.flags.DEFINE_string("headline_scope_name", "headline_var_scope", "Variable scope of Headline textsum model")

FLAGS = tf.app.flags.FLAGS

def read_data(source_path, target_path, max_size=None):
  """Read data from source and target files and put into buckets.
  
  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < buckets[n][0] and
      len(target) < buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set

def create_model(session, forward_only):
  """Create headline model and initialize or load parameters in session."""
  # dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  # dtype = tf.float32
  initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
  # Adding unique variable scope to model
  with tf.variable_scope(FLAGS.headline_scope_name, reuse=None, initializer=initializer):
    model = seq2seq_model.Seq2SeqModel(
        FLAGS.vocab_size,
        FLAGS.vocab_size,
        buckets,
        FLAGS.size,
        FLAGS.num_layers,
        FLAGS.max_gradient_norm,
        FLAGS.batch_size,
        FLAGS.learning_rate,
        FLAGS.learning_rate_decay_factor,
        use_lstm = True, # LSTM instend of GRU
        num_samples = FLAGS.num_samples,
        forward_only=forward_only)
  
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt:
    model_checkpoint_path = ckpt.model_checkpoint_path
    print("Reading model parameters from %s" % model_checkpoint_path)
    saver = tf.train.Saver()
    saver.restore(session, tf.train.latest_checkpoint(FLAGS.train_dir))
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
  
  return model

def train():
  # Prepare Headline data.
  print("Preparing Headline data in %s" % FLAGS.data_dir)
  src_train, dest_train, src_dev, dest_dev, _, _ = data_utils.prepare_headline_data(FLAGS.data_dir, FLAGS.vocab_size)
  
  # device config for CPU usage
  config = tf.ConfigProto(device_count={"CPU": 4}, # limit to 4 CPU usage
                   inter_op_parallelism_threads=1, 
                   intra_op_parallelism_threads=2) # n threads parallel for ops
  
  with tf.Session(config = config) as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, False)

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)
    dev_set = read_data(src_dev, dest_dev)
    train_set = read_data(src_train, dest_train, FLAGS.max_train_data_size)
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    trainbuckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    while True:
      # Choose a bucket according to data distribution. We pick a random number
      # in [0, 1] and use the corresponding interval in trainbuckets_scale.
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(trainbuckets_scale))
                       if trainbuckets_scale[i] > random_number_01])

      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          train_set, bucket_id)
      
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.train_dir, "headline_large.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        # Run evals on development set and print their perplexity.
        for bucket_id in xrange(len(buckets)):
          if len(dev_set[bucket_id]) == 0:
            print("  eval: empty bucket %d" % (bucket_id))
            continue
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              dev_set, bucket_id)
          _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
          eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
              "inf")
          print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
        sys.stdout.flush()

def main(_):
  train()

if __name__ == "__main__":
  tf.app.run()
