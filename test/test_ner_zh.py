#coding:utf-8

#Set Default codec coding to utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
print sys.getdefaultencoding()

import deepnlp.segmenter as segmenter
from deepnlp import ner_tagger
tagger = ner_tagger.load_model(lang = 'zh')

#Segmentation
text = "我爱吃北京烤鸭"
words = segmenter.seg(text.decode('utf-8'))
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
