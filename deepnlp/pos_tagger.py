#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
POS module global function 'predict'
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
pkg_path = os.path.dirname(os.path.abspath(__file__)) # .../deepnlp/
sys.path.append(pkg_path)
from pos import pos_model as pos_model
from pos import reader as pos_reader

class ModelLoader(object):
    
    def __init__(self, lang, data_path, ckpt_path):
        self.lang = lang
        self.data_path = data_path
        self.ckpt_path = ckpt_path
        print("Starting new Tensorflow session...")
        self.session = tf.Session()
        print("Initializing pos_tagger class...")
        self.model = self._init_pos_model(self.session, self.ckpt_path)

    def predict(self, words):
        '''
        Coding: utf-8 for Chinese Characters
        Return tuples of [(word, tag),...]
        '''
        tagging = self._predict_pos_tags(self.session, self.model, words, self.data_path)
        return tagging
    
    ## Initialize and Instance, Define Config Parameters for POS Tagger
    def _init_pos_model(self, session, ckpt_path):
        """Create POS Tagger model and initialize with random or load parameters in session."""
        # initilize config
        # config = POSConfig()   # Choose the config of language option
        config = pos_model.get_config(self.lang)
        config.batch_size = 1
        config.num_steps = 1 # iterator one token per time
      
        with tf.variable_scope("pos_var_scope"):  #Need to Change in Pos_Tagger Save Function
            model = pos_model.POSTagger(is_training=False, config=config) # save object after is_training
    
        if tf.gfile.Exists(ckpt_path):
            print("Loading model parameters from %s" % ckpt_path)

            all_vars = tf.all_variables()
            model_vars = [k for k in all_vars if k.name.startswith("pos_var_scope")]
            tf.train.Saver(model_vars).restore(session, ckpt_path)

        else:
            print("Model not found, created with fresh parameters.")
            session.run(tf.initialize_all_variables())
        return model
    
    def _predict_pos_tags(self, session, model, words, data_path):
        '''
        Define prediction function of POS Tagging
        return tuples [(word, tag)]
        '''
        word_data = pos_reader.sentence_to_word_ids(data_path, words)
        tag_data = [0]*len(word_data)
        state = session.run(model.initial_state)
        
        predict_id =[]
        for step, (x, y) in enumerate(pos_reader.iterator(word_data, tag_data, model.batch_size, model.num_steps)):
            #print ("Current Step" + str(step))
            fetches = [model.cost, model.final_state, model.logits]
            feed_dict = {}
            feed_dict[model.input_data] = x
            feed_dict[model.targets] = y
            for i, (c, h) in enumerate(model.initial_state):
              feed_dict[c] = state[i].c
              feed_dict[h] = state[i].h
            
            _, _, logits  = session.run(fetches, feed_dict)
            predict_id.append(int(np.argmax(logits)))    
            #print (logits)
        predict_tag = pos_reader.word_ids_to_sentence(data_path, predict_id)
        return zip(words, predict_tag)
    
def load_model(lang = 'zh'):
    data_path = os.path.join(pkg_path, "pos/data", lang) # POS vocabulary data path
    ckpt_path = os.path.join(pkg_path, "pos/ckpt", lang, "pos.ckpt") # POS model checkpoint path
    return ModelLoader(lang, data_path, ckpt_path)

