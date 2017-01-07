#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
text summarizer class for calling from other module
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import numpy as np
import tensorflow as tf

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # parent folder
sys.path.append(parent_dir)

from textsum import data_utils # absolute import
from textsum import seq2seq_model # absolute import
from textsum.headline import config
from textsum.headline import create_model
from textsum.headline import buckets

config # new Large Config, set to tf.app.flags
FLAGS = tf.app.flags.FLAGS # Load tf.FLAGS param from headline module
buckets = buckets

class ModelLoader(object):

    def __init__(self, lang):
        self.lang = lang
        print("Starting new Tensorflow session...")
        self.session = tf.Session()
        print("Initializing text summarization class...")
        self.model = self._init_model(self.session)
    
    def _init_model(self, session):
        """Create POS Tagger model and initialize with random or load parameters in session."""
        model = create_model(session, True)
        model.batch_size = FLAGS.batch_size
        return model
    
    def summarize(self, sentence):
        '''
        args: 
            sentence: space separated string of words
        return:
            headline_sum: space separated string of words
        '''
        # Load vocabularies.
        vocab_path = os.path.join(FLAGS.data_dir,"vocab")
        
        vocab, _ = data_utils.initialize_vocabulary(vocab_path)
        _, rev_vocab = data_utils.initialize_vocabulary(vocab_path)
        
        token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), vocab)
        print (token_ids) # print token ids
        # Which bucket does it belong to?
        bucket_id = min([b for b in xrange(len(buckets))
                       if buckets[b][0] > len(token_ids)])
        # Get a 1-element batch to feed the sentence to the model.
        print ("current bucket id" + str(bucket_id))
        encoder_inputs, decoder_inputs, target_weights = self.model.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id)
        _, _, output_logits_batch = self.model.step(self.session, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
        # This is a greedy decoder - outputs are just argmaxes of output_logits.      
        output_logits = []
        for item in output_logits_batch:
            output_logits.append(item[0])
        outputs = [int(np.argmax(logit)) for logit in output_logits]
        # outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
        # If there is an EOS symbol in outputs, cut them at that point.
        if data_utils.EOS_ID in outputs:
            outputs = outputs[:outputs.index(data_utils.EOS_ID)]
        # combining the str
        headline_sum = " ".join([tf.compat.as_str(rev_vocab[output]) for output in outputs])
        return headline_sum

def load_model(lang = 'zh'):
    return ModelLoader(lang)

