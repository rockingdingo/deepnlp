#coding:utf-8
from __future__ import unicode_literals # compatible with python3 unicode

from deepnlp import segmenter
from deepnlp import pos_tagger
tagger = pos_tagger.load_model(lang = 'zh')

#Segmentation
text = "我爱吃北京烤鸭"         # unicode coding, py2 and py3 compatible
words = segmenter.seg(text)
print(" ".join(words).encode('utf-8'))

#POS Tagging
tagging = tagger.predict(words)
for (w,t) in tagging:
    str = w + "/" + t
    print(str.encode('utf-8'))

#Results
#我/r
#爱/v
#吃/v
#北京/ns
#烤鸭/n
