#coding:utf-8

#Set Default codec coding to utf-8 to print chinese correctly
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
print sys.getdefaultencoding()

import deepnlp.segmenter as segmenter
from deepnlp import pos_tagger
tagger = pos_tagger.load_model(lang = 'zh')

#Segmentation
text = "我爱吃北京烤鸭"
words = segmenter.seg(text.decode('utf-8')) # words in unicode coding
print (" ".join(words).encode('utf-8'))

#POS Tagging
tagging = tagger.predict(words)
for (w,t) in tagging:
    str = w + "/" + t
    print (str.encode('utf-8'))

#Results
#我/r
#爱/v
#吃/v
#北京/ns
#烤鸭/n
