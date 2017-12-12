#!/usr/bin/python
# -*- coding:utf-8 -*-
# Reading NER data input_data and target_data

"""Utilities for reading NER train, dev and test files files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals # compatible with python3 unicode

import collections
import sys, os
import codecs
import re

import numpy as np
import tensorflow as tf

global UNKNOWN, DELIMITER
UNKNOWN = "*"
DELIMITER = "\s+" # line delimiter

def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return f.read().replace("\n", "").split()

def _read_file(filename):
  sentences = [] # list(list(str))
  words = []
  file = codecs.open(filename, encoding='utf-8')
  for line in file:
    wordsplit = re.split(DELIMITER, line.replace("\n",""))
    sentences.append(wordsplit) # list(list(str))
    words.extend(wordsplit) # list(str)
  return words, sentences

# input format word2/tag2 word2/tag2
def _split_word_tag(data):
    word = []
    tag = []
    for word_tag_pair in data:
        pairs = word_tag_pair.split("/")
        if (len(pairs)==2):
            # word or tag not equal to ""
            if (len(pairs[0].strip())!=0 and len(pairs[1].strip())!=0):
                word.append(pairs[0])
                tag.append(pairs[1])
    return word, tag

def _build_vocab(filename):
  words, sentences = _read_file(filename)
  word, tag = _split_word_tag(words)
  # split word and tag data
  
  # word dictionary
  word.append(UNKNOWN)
  counter_word = collections.Counter(word)
  count_pairs_word = sorted(counter_word.items(), key=lambda x: (-x[1], x[0]))

  wordlist, _ = list(zip(*count_pairs_word))
  word_to_id = dict(zip(wordlist, range(len(wordlist))))
  
  # tag dictionary
  tag.append(UNKNOWN)
  counter_tag = collections.Counter(tag)
  count_pairs_tag = sorted(counter_tag.items(), key=lambda x: (-x[1], x[0]))

  taglist, _ = list(zip(*count_pairs_tag))
  tag_to_id = dict(zip(taglist, range(len(taglist))))
  return word_to_id, tag_to_id

def _save_vocab(dict, path):
  # save utf-8 code dictionary
  file = codecs.open(path, "w", encoding='utf-8')
  for k, v in dict.items():
    # k is unicode, v is int
    line = k + "\t" + str(v) + "\n" # unicode
    file.write(line)

def _read_vocab(path):
  # read utf-8 code
  file = codecs.open(path, encoding='utf-8')
  vocab_dict = {}
  for line in file:
    pair = line.replace("\n","").split("\t")
    vocab_dict[pair[0]] = int(pair[1])
  return vocab_dict

def sentence_to_word_ids(data_path, words):
  word_to_id = _read_vocab(os.path.join(data_path, "word_to_id"))
  wordArray = [word_to_id[w] if w in word_to_id else word_to_id[UNKNOWN] for w in words]
  return wordArray

def word_ids_to_sentence(data_path, ids):
  tag_to_id = _read_vocab(os.path.join(data_path, "tag_to_id"))
  id_to_tag = {id:tag for tag, id in tag_to_id.items()}
  tagArray = [id_to_tag[i] if i in id_to_tag else id_to_tag[0] for i in ids]
  return tagArray

def _file_to_word_ids(filename, word_to_id, tag_to_id):
  words, sentences = _read_file(filename)
  word, tag = _split_word_tag(words)
  wordArray = [word_to_id[w] if w in word_to_id else word_to_id[UNKNOWN] for w in word]
  tagArray = [tag_to_id[t] if t in tag_to_id else tag_to_id[UNKNOWN] for t in tag]
  return wordArray, tagArray

def load_data(data_path=None):
  """Load NER raw data from data directory "data_path".
  Args: data_path
  Returns:
    tuple (train_data, valid_data, test_data, vocab_size)
    where each of the data objects can be passed to iterator.
  """

  train_path = os.path.join(data_path, "train.txt")
  dev_path = os.path.join(data_path, "dev.txt")
  test_path = os.path.join(data_path, "test.txt")
  
  word_to_id, tag_to_id = _build_vocab(train_path)
  # Save word_dict and tag_dict
  _save_vocab(word_to_id, os.path.join(data_path, "word_to_id"))
  _save_vocab(tag_to_id, os.path.join(data_path, "tag_to_id"))
  print ("word dictionary size "+ str(len(word_to_id)))
  print ("tag dictionary size "+ str(len(tag_to_id)))
  
  train_word, train_tag = _file_to_word_ids(train_path, word_to_id, tag_to_id)
  print ("train dataset: "+ str(len(train_word)) + " " + str(len(train_tag)))
  dev_word, dev_tag = _file_to_word_ids(dev_path, word_to_id, tag_to_id)
  print ("dev dataset: "+ str(len(dev_word)) + " " + str(len(dev_tag)))
  test_word, test_tag = _file_to_word_ids(test_path, word_to_id, tag_to_id)
  print ("test dataset: "+ str(len(test_word)) + " " + str(len(test_tag)))
  vocab_size = len(word_to_id)
  return (train_word, train_tag, dev_word, dev_tag, test_word, test_tag, vocab_size)

def iterator(word_data, tag_data, batch_size, num_steps):
  """Iterate on the raw NER tagging file data.
  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.

  Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one.

  Raises:
    ValueError: if batch_size or num_steps are too high.
  """
  word_data = np.array(word_data, dtype=np.int32)
  
  data_len = len(word_data)
  batch_len = data_len // batch_size
  xArray = np.zeros([batch_size, batch_len], dtype=np.int32)
  yArray = np.zeros([batch_size, batch_len], dtype=np.int32)
  for i in range(batch_size):
    xArray[i] = word_data[batch_len * i:batch_len * (i + 1)]
    yArray[i] = tag_data[batch_len * i:batch_len * (i + 1)]
    
  # how many epoch to finish all samples, 
  # language model (batch_len-1) y is offset by 1
  # sequence tagging batch_len
  epoch_size = (batch_len) // num_steps
  
  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    x = xArray[:, i*num_steps:(i+1)*num_steps]
    # for language model, {x: X(t), y: X(t+1)}
    # for sequence tagging, {x: X(t), y: Y(t)}
    y = yArray[:, i*num_steps:(i+1)*num_steps]
    yield (x, y)

def main():
  """
  Test load_data method and iterator method
  """
  data_path ="/mnt/pypi/deepnlp/deepnlp/ner/data/zh"
  print ("Data Path: " + data_path)
  train_word, train_tag, dev_word, dev_tag, test_word, test_tag, _ = load_data(data_path)
  
  iter = iterator(train_word, train_tag, 1, 35)
  count = 0
  for step, (x, y) in enumerate(iter):
    count +=1;
    #rint (x)
    #rint (y)
  print ("Count:" + str(count))
  
  ids = [1,4,2,7,20]
  sentence = word_ids_to_sentence(data_path, ids)
  print (sentence)

if __name__ == '__main__':
  main()
