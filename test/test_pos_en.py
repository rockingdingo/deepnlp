#coding:utf-8
from __future__ import unicode_literals

import deepnlp
deepnlp.download(module='pos',name='en')                     # download the POS pretrained models from github if installed from pip

from deepnlp import pos_tagger
tagger = pos_tagger.load_model(name = 'en')  # Loading English model, lang code 'en'

#Segmentation
text = "I want to see a funny movie"
words = text.split(" ")
print (" ".join(words))

#POS Tagging
tagging = tagger.predict(words)
for (w,t) in tagging:
    pair = w + "/" + t
    print (pair)

#Results
#I/nn
#want/vb
#to/to
#see/vb
#a/at
#funny/jj
#movie/nn
