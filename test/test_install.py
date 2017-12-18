#coding=utf-8
from __future__ import unicode_literals

import tensorflow as tf
import deepnlp

# Download module and domain-specific model
deepnlp.download(module = 'segment', name = 'zh_entertainment')
deepnlp.download(module = 'pos', name = 'en') 
deepnlp.download(module = 'ner', name = 'zh_o2o')

# Download module
deepnlp.download('segment')
deepnlp.download('pos')
deepnlp.download('ner')
deepnlp.download('parse')

# deepnlp.download()

## 测试 load model
from deepnlp import segmenter
try:
    tokenizer = segmenter.load_model(name = 'zh')
    tokenizer = segmenter.load_model(name = 'zh_o2o')
    tokenizer = segmenter.load_model(name = 'zh_entertainment')
except Exception as e:
    print ("DEBUG: ERROR Found...")    
    print (e)

## pos
from deepnlp import pos_tagger
try:
    tagger = pos_tagger.load_model(name = 'en')  # Loading English model, lang code 'en'
    tagger = pos_tagger.load_model(name = 'zh')  # Loading English model, lang code 'en'
except Exception as e:
    print ("DEBUG: ERROR Found...")
    print (e)


## ner
from deepnlp import ner_tagger
try:
    my_tagger = ner_tagger.load_model(name = 'zh')
    my_tagger = ner_tagger.load_model(name = 'zh_o2o')
    my_tagger = ner_tagger.load_model(name = 'zh_entertainment')
except Exception as e:
    print ("DEBUG: ERROR Found...")    
    print (e)

## parse
from deepnlp import nn_parser
try:
    parser = nn_parser.load_model(name = 'zh')
except Exception as e:
    print ("DEBUG: ERROR Found...")    
    print (e)

