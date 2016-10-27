#coding:utf-8

#Set Default codec coding to utf-8 to print chinese correctly
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
print sys.getdefaultencoding()

import deepnlp.segmenter as segmenter
import deepnlp.pos_tagger as pos_tagger

#Segmentation
text = "我爱吃北京烤鸭"
words = segmenter.seg(text.decode('utf-8'))
print (" ".join(words).encode('utf-8'))

#POS Tagging
tagging = pos_tagger.predict(words)
for (w,t) in tagging:
    str = w + "/" + t
    print (str.encode('utf-8'))

#Results
#我/r
#爱/v
#吃/v
#北京/ns
#烤鸭/n
