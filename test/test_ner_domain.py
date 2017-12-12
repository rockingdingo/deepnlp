#!/usr/bin/python
# -*- coding:utf-8 -*-

from __future__ import unicode_literals # compatible with python3 unicode

import deepnlp
deepnlp.download('ner')  # download the NER pretrained models from github if installed from pip

from deepnlp import ner_tagger

# Example 1. Change to other dict
tagger = ner_tagger.load_model(name = 'zh_o2o')   # Base LSTM Based Model + zh_o2o dictionary
text = "北京 望京 最好吃 的 小龙虾 在 哪里"
words = text.split(" ")
tagging = tagger.predict(words, tagset = ['city', 'area', 'dish'])
for (w,t) in tagging:
    pair = w + "/" + t
    print (pair)

#Result
#北京/city
#望京/area
#最好吃/nt
#的/nt
#小/nt
#龙虾/dish
#在/nt
#哪里/nt

# Example 2: Switch to base NER LSTM-Based Model and domain-specific dictionary
tagger = ner_tagger.load_model(name = 'zh')   # Base LSTM Based Model
#Load Entertainment Dict
tagger.load_dict("zh_entertainment")
text = "你 最近 在 看 胡歌 演的 猎场 吗 ?"
words = text.split(" ")
tagging = tagger.predict(words, tagset = ['actor'])
for (w,t) in tagging:
    pair = w + "/" + t
    print (pair)

#Result
#你最近/nt
#在/nt
#看/nt
#胡歌/actor
#演的/nt
#猎场/nt
#吗/nt
#?/nt


# Load User Defined Dict
# user_dict_path = "./data/entity_tags.dic"
# tagger.load_user_dict(user_dict_path)


