#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
@author: xichen ding
@date: 2016-11-15
@rev: 2017-11-01
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals # compatible with python3 unicode coding

import sys, os
import tensorflow as tf
import numpy as np
import glob

# adding pos submodule to sys.path, compatible with py3 absolute_import
pkg_path = os.path.dirname(os.path.abspath(__file__)) # .../deepnlp/
sys.path.append(pkg_path)
from pos import pos_model as pos_model
from pos import reader as pos_reader
from model_util import get_model_var_scope
from model_util import _pos_scope_name
from model_util import registered_models

class ModelLoader(object):
    
    def __init__(self, name, data_path, ckpt_path):
        self.name = name   # model name
        self.data_path = data_path
        self.ckpt_path = ckpt_path  # the path of the ckpt file, e.g. ./ckpt/zh/pos.ckpt
        print("NOTICE: Starting new Tensorflow session...")
        self.session = tf.Session()
        print("NOTICE: Initializing pos_tagger class...")
        self.model = None
        self.var_scope = _pos_scope_name
        self._init_pos_model(self.session, self.ckpt_path)  # Initialization model

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
        config = pos_model.get_config(self.name)
        config.batch_size = 1
        config.num_steps = 1 # iterator one token per time
        model_var_scope = get_model_var_scope(self.var_scope, self.name)
        print ("NOTICE: Input POS Model Var Scope Name '%s'" % model_var_scope)
        # Check if self.model already exist
        if self.model is None:
            with tf.variable_scope(model_var_scope, tf.AUTO_REUSE):
                self.model = pos_model.POSTagger(is_training=False, config=config) # save object after is_training
        # Load Specific .data* ckpt file
        if len(glob.glob(ckpt_path + '.data*')) > 0: # file exist with pattern: 'pos.ckpt.data*'
            print("NOTICE: Loading model parameters from %s" % ckpt_path)
            all_vars = tf.global_variables()
            model_vars = [k for k in all_vars if model_var_scope in k.name.split("/")]
            tf.train.Saver(model_vars).restore(session, ckpt_path)
        else:
            print("NOTICE: Model not found, Try to run method: deepnlp.download(module='pos', name='%s')" % self.name)
            print("NOTICE: Created with fresh parameters.")
            session.run(tf.global_variables_initializer())
    
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

def load_model(name = 'zh'):
    ''' data_path e.g.: ./deepnlp/pos/data/zh
        ckpt_path e.g.: ./deepnlp/pos/ckpt/zh/pos.ckpt
        ckpt_file e.g.: ./deepnlp/pos/ckpt/zh/pos.ckpt.data-00000-of-00001
    '''
    registered_model_list = registered_models[0]['pos']
    if name not in registered_model_list:
        print ("WARNING: Input model name '%s' is not registered..." % name)
        print ("WARNING: Please register the name in model_util.registered_models...")
        return None
    data_path = os.path.join(pkg_path, "pos/data", name) # POS vocabulary data path
    ckpt_path = os.path.join(pkg_path, "pos/ckpt", name, "pos.ckpt") # POS model checkpoint path
    return ModelLoader(name, data_path, ckpt_path)
