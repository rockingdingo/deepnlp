#coding:utf-8
from __future__ import unicode_literals

from deepnlp import pos_tagger
tagger = pos_tagger.load_model(lang = 'en')  # Loading English model, lang code 'en'

#Segmentation
text = "I want to see a funny movie"
words = text.split(" ")
print (" ".join(words).encode('utf-8'))

#POS Tagging
tagging = tagger.predict(words)
for (w,t) in tagging:
    str = w + "/" + t
    print (str.encode('utf-8'))

#Results
#I/nn
#want/vb
#to/to
#see/vb
#a/at
#funny/jj
#movie/nn
