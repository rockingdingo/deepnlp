#coding:utf-8
from __future__ import unicode_literals # compatible with python3 unicode

from deepnlp import segmenter
from deepnlp import ner_tagger
tagger = ner_tagger.load_model(lang = 'zh')

#Segmentation
text = "我爱吃北京烤鸭"
words = segmenter.seg(text)
print (" ".join(words).encode('utf-8'))

#NER tagging
tagging = tagger.predict(words)
for (w,t) in tagging:
    str = w + "/" + t
    print (str.encode('utf-8'))

#Results
#我/nt
#爱/nt
#吃/nt
#北京/p
#烤鸭/nt
