#!/usr/bin/python
# -*- coding:utf-8 -*-

'''
NER module global function 'predict'
@author: xichen ding
@date: 2016-10-27
'''

import sys, os
import tensorflow as tf
import numpy as np

reload(sys)
sys.setdefaultencoding('utf-8')

from ner.ner_model import LargeConfig as nerConfig # Tensorflow LSTM model parameters config
from ner.ner_model import NERTagger
import ner.reader as ner_reader

import deepnlp
pkg_path = (deepnlp.__path__)[0]
ner_data_path = os.path.join(pkg_path, "ner/data") # NER vocabulary data path
ner_ckpt_path = os.path.join(pkg_path, "ner/ckpt/ner.ckpt") # NER model ckpt path

class ModelLoader(object):

    def __init__(self):
        print("Starting new Tensorflow session...")
        self.session = tf.Session()
        print("Initializing ner_tagger model...")
        self.ner_model = self._init_ner_model(self.session, ner_ckpt_path)

    def predict(self, words):
        ''' 
        Coding: utf-8 for Chinese Characters
        Return tuples of [(word, tag),...]
        '''
        tagging = self._predict_ner_tags(self.session, self.ner_model, words, ner_data_path)
        return tagging
    
    ## Define Config Parameters for NER Tagger
    def _init_ner_model(self, session, ckpt_path):
        """Create ner Tagger model and initialize or load parameters in session."""
        # initilize config
        config = nerConfig()
        config.batch_size = 1
        config.num_steps = 1 # iterator one token per time
      
        with tf.variable_scope("ner_var_scope"):
            model = NERTagger(is_training=True, config=config) # save object after is_training
    
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

# Create Instance of Model Loader
ml = ModelLoader()

# Global Function
predict = ml.predict
