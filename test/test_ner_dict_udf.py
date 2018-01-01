#!/usr/bin/python
# -*- coding:utf-8 -*-

from __future__ import unicode_literals # compatible with python3 unicode

import deepnlp
deepnlp.download('ner')  # download the NER pretrained models from github if installed from pip

from deepnlp import ner_tagger
tagger = ner_tagger.load_model(name = 'zh')    # Base LSTM Based Model
tagger.load_dict("zh_o2o")                     # Change to other dict

text = "北京 望京 最好吃 的 黑椒 牛排 在哪里"
words = text.split(" ")

# Use the prefix dict and merging function to combine separated words
tagging = tagger._predict_ner_tags_dict(words, merge = True)
print ("DEBUG: NER tagger zh_o2o dictionary")
for (w,t) in tagging:
    pair = w + "/" + t
    print (pair)

#北京/city
#望京/area
#最好吃/nt
#的/nt
#黑椒牛排/dish
#在哪里/nt


# Word Sense Disambuguition
# Sometime each word have multiple tags, use can define customized UDF(user_difine_function)
tagger.load_dict("zh_entertainment")                     # Change to other dict
text = "今天 我 看 了 琅 琊 榜"
words = text.split(" ")
tagging = tagger._predict_ner_tags_dict(words, merge = True)
print ("DEBUG: NER tagger zh_entertainment dictionary with 'merge = True'")
for (w,t) in tagging:
    pair = w + "/" + t
    print (pair)

# '琅琊榜' have two category: 'list_name' and 'teleplay'
# Disambiguation
from deepnlp.ner_tagger import udf_disambiguation_cooccur
from deepnlp.ner_tagger import udf_default

word = "琅琊榜"
tags = ['list_name', 'teleplay']
context = ["今天", "我", "看", "了"]
tag_feat_dict = {}
# Most Freq Word Feature of two tags
tag_feat_dict['list_name'] = ['听', '专辑', '音乐']
tag_feat_dict['teleplay'] = ['看', '电视', '影视']
# Disambuguiation Prob
tag, prob = udf_disambiguation_cooccur(word, tags, context, tag_feat_dict)
print ("DEBUG: NER tagger zh_entertainment with user defined function for disambuguiation")
print ("Word:%s, Tag:%s, Prob:%f" % (word, tag, prob))

# Combine the results and load the udfs
tagging = tagger._predict_ner_tags_dict(words, merge = True, udfs = [udf_disambiguation_cooccur])
for (w,t) in tagging:
    pair = w + "/" + t
    print (pair)
