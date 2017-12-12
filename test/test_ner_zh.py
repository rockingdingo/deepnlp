#coding:utf-8
from __future__ import unicode_literals # compatible with python3 unicode

import deepnlp
deepnlp.download('ner')  # download the NER pretrained models from github if installed from pip

from deepnlp import segmenter
from deepnlp import ner_tagger

tokenizer = segmenter.load_model(name = 'zh')
tagger = ner_tagger.load_model(name = 'zh')

#Segmentation
text = "我爱吃北京烤鸭"
words = tokenizer.seg(text)
print (" ".join(words))

#NER tagging
tagging = tagger.predict(words)
for (w,t) in tagging:
    pair = w + "/" + t
    print (pair)

#Results
#我/nt
#爱/nt
#吃/nt
#北京/city
#烤鸭/nt
