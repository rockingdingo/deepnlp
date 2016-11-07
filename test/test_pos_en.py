#coding:utf-8

#Set Default codec coding to utf-8 to print chinese correctly
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
print sys.getdefaultencoding()

import deepnlp.segmenter as segmenter
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
