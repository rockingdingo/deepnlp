#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
NER module global function 'predict'
@author: xichen ding
@date: 2016-11-15
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals # compatible with python3 unicode coding

import sys, os
import tensorflow as tf
import numpy as np

# adding pos submodule to sys.path, compatible with py3 absolute_import
pkg_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(pkg_path)

from ner import ner_model as ner_model
from ner import reader as ner_reader

class ModelLoader(object):

    def __init__(self, lang, data_path, ckpt_path):
        self.lang = lang
        self.data_path = data_path
        self.ckpt_path = ckpt_path
        print("Starting new Tensorflow session...")
        self.session = tf.Session()
        print("Initializing ner_tagger model...")
        self.model = self._init_ner_model(self.session, self.ckpt_path)
    
    def predict(self, words):
        ''' 
        Coding: utf-8 for Chinese Characters
        Return tuples of [(word, tag),...]
        '''
        tagging = self._predict_ner_tags(self.session, self.model, words, self.data_path)
        return tagging
    
    ## Define Config Parameters for NER Tagger
    def _init_ner_model(self, session, ckpt_path):
        """Create ner Tagger model and initialize or load parameters in session."""
        # initilize config
        config = ner_model.get_config(self.lang)
        config.batch_size = 1
        config.num_steps = 1 # iterator one token per time
        
        with tf.variable_scope("ner_var_scope"):
            model = ner_model.NERTagger(is_training=True, config=config) # save object after is_training
    
        if tf.gfile.Exists(ckpt_path):
            print("Loading model parameters from %s" % ckpt_path)

            all_vars = tf.all_variables()
            model_vars = [k for k in all_vars if k.name.startswith("ner_var_scope")]
            tf.train.Saver(model_vars).restore(session, ckpt_path)

        else:
            print("Model not found, created with fresh parameters.")
            session.run(tf.initialize_all_variables())
        return model
    
    def _predict_ner_tags(self, session, model, words, data_path):
        '''
        Define prediction function of ner Tagging
        return tuples (word, tag)
        '''
        word_data = ner_reader.sentence_to_word_ids(data_path, words)
        tag_data = [0]*len(word_data)
        state = session.run(model.initial_state)
        
        predict_id =[]
        for step, (x, y) in enumerate(ner_reader.iterator(word_data, tag_data, model.batch_size, model.num_steps)):
            fetches = [model.cost, model.final_state, model.logits]
            feed_dict = {}
            feed_dict[model.input_data] = x
            feed_dict[model.targets] = y
            for i, (c, h) in enumerate(model.initial_state):
              feed_dict[c] = state[i].c
              feed_dict[h] = state[i].h
            
            _, _, logits  = session.run(fetches, feed_dict)
            predict_id.append(int(np.argmax(logits)))    
        predict_tag = ner_reader.word_ids_to_sentence(data_path, predict_id)
        return zip(words, predict_tag)

def load_model(lang = 'zh'):
    data_path = os.path.join(pkg_path, "ner/data", lang) # NER vocabulary data path
    ckpt_path = os.path.join(pkg_path, "ner/ckpt", lang, "ner.ckpt") # NER model checkpoint path
    return ModelLoader(lang, data_path, ckpt_path)

