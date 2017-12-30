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
import pickle

from dict_util import gen_prefix_dict

# adding pos submodule to sys.path, compatible with py3 absolute_import
pkg_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(pkg_path)

from ner import ner_model as ner_model
from ner import reader as ner_reader
from model_util import get_model_var_scope
from model_util import _ner_scope_name
from model_util import registered_models

### Define Constant
TAG_NONE_ENTITY = "nt"

ENTITY_TAG_DICT = "entity_tags.dic"

ENTITY_TAG_DICT_PICKLE = "entity_tags.dic.pkl"

DEFAULT_DICT_ZH_NAME = "zh"

# User Define Function Disambiguation
def udf_default(word, tags, *args):
    """ Default get the first tag
        Return: tag, confidence
    """
    if (len(tags) > 0):
        return tags[0], 1.0
    else:
        return TAG_NONE_ENTITY
    return tags[0], 1.0

def udf_disambiguation_cooccur(word, tags, context, tag_feat_dict, *args):
    """ Disambiguation based on cooccurence of context and tag_feat_dict
        Args: word: input word
            tags: multiple tags on word
            context: list of words surrounding current word of a window
    """
    if (len(tag_feat_dict) == 0) or (len(tags) == 0):
        return None, 0.0

    num = len(tags)
    coocur_dict = {}
    coocur_count = []
    for tag in tags:
        feat_words = tag_feat_dict[tag] if tag in tag_feat_dict else []
        common = []
        for feat in feat_words:
            if feat in context:
                common.append(feat)
        coocur_dict[tag] = len(common)  # How many occurence under current tags
        coocur_count.append(len(common))
    vec = np.array(coocur_count)
    total = np.sum(vec)
    prob_vec = []
    if total > 0.0:
        prob_vec = vec/total
    else:
        prob_vec = 0.0 * vec
    max_index = np.argmax(prob_vec)
    return tags[max_index], prob_vec[max_index]

def ensemble_udf(udfs, word, tags, *args):
    """ Embed Multiple UDFs to get the ambiguation tag
    """
    tag_count_dict = {}
    for udf in udfs:
        tag, confidence = udf(word, tags, *args)
        if tag is not None:
            if tag in tag_count_dict:
                tag_count_dict[tag] = tag_count_dict[tag] + 1
            else:
                tag_count_dict[tag] = 1
    max_cnt = -1
    max_cnt_tag = TAG_NONE_ENTITY
    for tag in tag_count_dict.keys():
        cur_cnt = tag_count_dict[tag]
        if (cur_cnt > max_cnt):
            max_cnt = cur_cnt
            max_cnt_tag = tag
    return max_cnt_tag

class ModelLoader(object):

    def __init__(self, name, data_path, ckpt_path):
        self.name = name
        self.data_path = data_path
        self.ckpt_path = ckpt_path
        self.model_config_path = os.path.join(os.path.dirname(data_path), "models.conf")  #./data/models.conf
        print("NOTICE: Starting new Tensorflow session...")
        print("NOTICE: Initializing ner_tagger model...")
        self.session = tf.Session()
        self.model = None
        self.var_scope = _ner_scope_name
        self._init_ner_model(self.session, self.data_path, self.ckpt_path)  # Initialization model
        self.__prefix_dict = {}                             # private member variable
        self._load_dict(name)                               # load model dict zh + dict_name
    
    def predict(self, words, tagset = []):
        ''' 
            Args: words: list of string
                tagset: tags that is included in the final output, default [] means return all tags
            Return tuples of [(word, tag),...]
        '''
        model_tagging = self._predict_ner_tags_model(self.session, self.model, words, self.data_path)
        dict_tagging = self._predict_ner_tags_dict(words, merge = True, tagset = tagset, udfs = [udf_default])
        merge_tagging = self._merge_tagging(model_tagging, dict_tagging)
        return dict_tagging
    
    ## Define Config Parameters for NER Tagger
    def _init_ner_model(self, session, data_path, ckpt_path):
        """Create ner Tagger model and initialize or load parameters in session."""
        # initilize config
        config_dict = ner_reader.load_config(self.model_config_path)
        config = ner_model.get_config(config_dict, self.name)
        if config is None:
            print ("WARNING: Input model name %s has no configuration..." % self.name)
        config.batch_size = 1
        config.num_steps = 1 # iterator one token per time
        model_var_scope = get_model_var_scope(self.var_scope, self.name)
        print ("NOTICE: Input NER Model Var Scope Name '%s'" % model_var_scope)
        # Check if self.model already exist
        if self.model is None:
            with tf.variable_scope(model_var_scope, reuse = tf.AUTO_REUSE):
                self.model = ner_model.NERTagger(is_training=True, config=config) # save object after is_training
        #else:   # Model Graph Def already exist
        #    print ("DEBUG: Model Def already exists")
        # update model parameters
        if len(glob.glob(ckpt_path + '.data*')) > 0: # file exist with pattern: 'ner.ckpt.data*'
            print("NOTICE: Loading model parameters from %s" % ckpt_path)
            all_vars = tf.global_variables()
            model_vars = [k for k in all_vars if model_var_scope in k.name.split("/")]   # e.g. ner_var_scope_zh
            tf.train.Saver(model_vars).restore(session, ckpt_path)
        else:
            print("NOTICE: Model not found, Try to run method: deepnlp.download(module='ner', name='%s')" % self.name)
            print("NOTICE: Created with fresh parameters.")
            session.run(tf.global_variables_initializer())
        
    def _predict_ner_tags_model(self, session, model, words, data_path):
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
        predict_taggedwords = list(zip(words, predict_tag))
        return predict_taggedwords
    
    # internal variable for disambiguation
    tag_feat_dict = {}
    def set_tag_feat_dict(self, tag_feat_dict):
        self.tag_feat_dict = tag_feat_dict

    def _predict_ner_tags_dict(self, words, merge = False, tagset = [],udfs = [udf_default]):
        """ search NER tags from the whole sentences with Maximum Length
            Args: words: list of string 
                  merge: boolean , if merge the segmentation results 
                  udfs:  list of user defined functions
        """
        words_merge = self._preprocess_segment(words) if merge else words
        tokens = []
        include_all = True if len(tagset) == 0 else False
        for i in range(len(words_merge)):
            word = words_merge[i]
            if word in self.__prefix_dict:
                tags = self.__prefix_dict[word]
                if (tags):
                    # tag = tags[0]   # To Do, Add Disambiguity function
                    context = self._get_context_words(words_merge, i)                  # Getting surround context words
                    tag = ensemble_udf(udfs, word, tags, context, self.tag_feat_dict)  # Using Coocurrence to disambiguation
                    # Check if current tags is included in tagset
                    if (include_all):
                        tokens.append((word, tag))
                    else:
                        if tag in tagset:
                            tokens.append((word, tag))
                        else:
                            tokens.append((word, TAG_NONE_ENTITY))
                else:
                    tokens.append((word, TAG_NONE_ENTITY))
            else:
                tokens.append((word, TAG_NONE_ENTITY))
        return tokens
    
    def _get_context_words(self, words, i, window = 4):
        """ Get context words: a list of words within window of the given word
        """
        if (i >= len(words)):
            return None
        else:
            token_num = len(words)
            start_id = max(i - window, 0)
            end_id = min(i + window, (token_num - 1))
            context = words[start_id:i] + words[(i+1):(end_id+1)]
            return context
    
    # default setting
    max_iter = 1000
    def _preprocess_segment(self, words):
        """ Consolidate Words Segmentation and Merge words to get Maximum Length Segment
        """
        token_num = len(words)
        start_id = 0
        words_new = []
        lineno = 0
        while (start_id < token_num and lineno < self.max_iter):
            lineno += 1
            #print ("Start Id... %d " % start_id)
            step = 0
            # Get Boundry
            while(step < (token_num - start_id)):
                segment = "".join(words[start_id:(start_id + step + 1)])
                if segment in self.__prefix_dict:
                    step += 1
                else:
                    break
            # Check if current word is in Dict or Not
            if (step == 0):    # Current Word is not in Dictionary
                segment = "".join(words[start_id:(start_id + 1)])  # Current Word
                words_new.append(segment)
                #print ("Current Segment %s with step %d" % (segment, step))
                start_id += 1
            else:              # At least one word is in Dictionary
                segment = "".join(words[start_id:(start_id + step)])     #
                words_new.append(segment)
                #print ("Current Segment %s with step %d" % (segment, step))
                start_id += step
        return words_new

    def _merge_tagging(self, model_tagging, dict_tagging):
        """ Merge tagging results of model and dict
        """
        if (len(model_tagging) != len(dict_tagging)):
            print ("WARNING: Model Tagging Sequence and Dict Tagging Sequence are different")
            return None
        num = len(model_tagging)
        merge_tagging = []
        for i in range(num):
            word = model_tagging[i][0]
            model_tag = model_tagging[i][1]
            dict_tag = dict_tagging[i][1]
            if (dict_tag):  # not None
                merge_tagging.append((word, dict_tag))
            else:
                merge_tagging.append((word, model_tag))
        return merge_tagging
        
    def _load_default_dict(self, name = 'zh'):
        ''' internal method to load new default prefict_dict
            default dict
        '''
        print ("NOTICE: Start Loading Default Entity Tag Dictionary: %s ..." % name)
        default_dict_pickle_path = os.path.join(pkg_path, "ner/dict/", name, ENTITY_TAG_DICT_PICKLE)
        if not os.path.exists(default_dict_pickle_path):
            print ("ERROR: Input Pickle file doesn't exist:%s ..." % default_dict_pickle_path)
            return
        else:
            fr = open(default_dict_pickle_path, 'rb')
            try:
                self.__prefix_dict = pickle.load(fr)   # update to new dictionary
                print ("NOTICE: Loading NER Tagger Prefix Dict successfully, Dict Size: %d ..." % len(self.__prefix_dict))
            except:
                print ("ERROR: Failed to load pickle file %s" % default_dict_pickle_path)

    def _load_dict(self, dict_name):
        """ internal method during initialization
            load default zh dict and update input dict if necessary
            default dict zh + input dict
        """
        self.__prefix_dict = {}                         # empty current dict
        self._load_default_dict(DEFAULT_DICT_ZH_NAME)   # adding base dictionary
        if (dict_name != DEFAULT_DICT_ZH_NAME):         # load 'zh' + dict_name
            # add new input dict
            dict_path = os.path.join(pkg_path, "ner/dict/", dict_name, ENTITY_TAG_DICT_PICKLE)
            if os.path.exists(dict_path):
                #pdict = gen_prefix_dict(dict_path)
                pdict = {}
                fr = open(dict_path, 'rb')
                try:
                    pdict = pickle.load(fr)
                    print ("NOTICE: Loading Entity Tags Prefix Dict successfully, Dict Size: %d ..." % len(pdict))
                except:
                    print ("ERROR: Failed to load pickle file %s" % dict_path)
                # Check Entity Tag Conflicts
                pdict_clean = {}  # Check Dict Conflict: Newly Loaded Dict Have confict with default ones
                pdict_confict = {}
                for k in pdict.keys():
                    if k not in self.__prefix_dict:
                        pdict_clean[k] = pdict[k]
                    else:
                        pdict_confict[k] = pdict[k]
                if (len(pdict_confict) > 0):
                    print ("WARNING: Newly loaded dict have conflict with default dict, Size: %d" % len(pdict_confict))
                self.__prefix_dict.update(pdict_clean) # Merge Newly Added Dict to Default Dict
                print ("NOTICE: Loading Dictionary Successfully...")
            else:
                print ("ERROR: Dict Path doesn't exist: %s" % dict_path)

    def load_dict(self, dict_name):
        """ public method to load dict_name in the package
            default dict zh + input dict
        """
        # add default dict
        self._load_dict(dict_name)

    def load_user_dict(self, path):
        """ public method to load user defined ner slots dictionary
        """
        if os.path.exists(path):
            pdict = gen_prefix_dict(path)
            pdict_clean = {}  # Check Dict Conflict: Newly Loaded Dict Have confict with default ones
            pdict_confict = {}
            for k in pdict.keys():
                if k not in self.__prefix_dict:
                    pdict_clean[k] = pdict[k]
                else:
                    pdict_confict[k] = pdict[k]
            if (len(pdict_confict) > 0):
                print ("WARNING: Newly loaded dict have conflict with default dict: %d" % len(pdict_confict))
            self.__prefix_dict.update(pdict_clean) # Merge Newly Added Dict to Default Dict
            print ("NOTICE: Loading Dictionary Successfully...")
        else:
            print ("ERROR: User Dict Path doesn't exist: %s" % path)

def load_model(name = 'zh'):
    ''' Args: name: model name;
        data_path e.g.: ./deepnlp/ner/data/zh
        ckpt_path e.g.: ./deepnlp/ner/ckpt/zh/ner.ckpt
        ckpt_file e.g.: ./deepnlp/ner/ckpt/zh/ner.ckpt.data-00000-of-00001
    '''
    registered_model_list = registered_models[0]['ner']
    if name not in registered_model_list:
        print ("WARNING: Input model name '%s' is not registered..." % name)
        print ("WARNING: Please register the name in model_util.registered_models...")
        return None
    data_path = os.path.join(pkg_path, "ner/data", name) # NER vocabulary data path
    ckpt_path = os.path.join(pkg_path, "ner/ckpt", name, "ner.ckpt") # NER model checkpoint path
    return ModelLoader(name, data_path, ckpt_path)
