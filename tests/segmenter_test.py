#coding:utf-8

#Set Default codec coding to utf-8 to print chinese correctly
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
print sys.getdefaultencoding()

#Example for Sentences Segmentation
import deepnlp.segmenter as segmenter

text = "我爱吃北京烤鸭"
segList = segmenter.seg(text.decode('utf-8')) # python 2: function input: unicode, return unicode
text_seg = " ".join(segList)

print (text.encode('utf-8'))
print (text_seg.encode('utf-8'))
