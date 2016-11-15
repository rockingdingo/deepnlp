#coding:utf-8
from __future__ import unicode_literals # compatible with python3 unicode

#Example for Sentences Segmentation
from deepnlp import segmenter

text = "我爱吃北京烤鸭"
segList = segmenter.seg(text) # python 2/3: function input: unicode, return unicode
text_seg = " ".join(segList)

print (text.encode('utf-8'))
print (text_seg.encode('utf-8'))
