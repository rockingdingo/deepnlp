#!/usr/bin/python
# -*- coding:utf-8 -*-

'''
Main Pipiline Access for Deepnlp modules, Loading all pre-trained models during initilization
@author: xichen ding
@date: 2016/10/9
'''

import sys, os
import tensorflow as tf
import numpy as np
import CRFPP  # CRFPP for segmenting

reload(sys)
sys.setdefaultencoding('utf-8')

import segmenter as segmenter     # segmenter module
import pos_tagger as pos_tagger   # pos tagger
import ner_tagger as ner_tagger   # ner tagger

import deepnlp
pkg_path = (deepnlp.__path__)[0]

class Pipeline(object):

  def __init__(self, lang):
    print("Starting new Tensorflow session...")
    self.session = tf.Session()
    print("Loading pipeline modules...")
    self.tagger_pos = pos_tagger.load_model(lang) # class tagger_pos
    self.tagger_ner = ner_tagger.load_model(lang) # class tagger_ner
    
  def analyze(self, string):
    '''Return a list of three string output: segment, pos, ner'''
    res = []
    #segment
    words = segmenter.seg(string)
    segment_str = " ".join(words)
    res.append(segment_str)
    
    #POS
    pos_tagging = self.tag_pos(words)
    res.append(_concat_tuples(pos_tagging))
    
    #NER
    ner_tagging = self.tag_ner(words)
    res.append(_concat_tuples(ner_tagging))
    return res
  
  def segment(self, string):
    ''' Return list of [word]'''
    words = segmenter.seg(string)
    return words
  
  def tag_pos(self, words):
    ''' Return tuples of [(word, tag), ...]'''
    tagging = self.tagger_pos.predict(words)
    return tagging

  def tag_ner(self, words):
    ''' Return tuples of [(word, tag), ...]'''
    tagging = self.tagger_ner.predict(words)
    return tagging
  
  def parse(self, words):
    '''To Do'''
    return 0

def _concat_tuples(tagging):
  '''
  param: tuples of [(word, tag), ...]
  return: string, 'w1/t1 w2/t2 ...'
  '''
  TOKEN_BLANK = " "
  wl = [] # wordlist
  for (x, y) in tagging:
    wl.append(str(x + "/" + y))
  concat_str = TOKEN_BLANK.join(wl)
  return concat_str

def load_model(lang = 'zh'):
  return Pipeline(lang)

