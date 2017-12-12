#coding=utf-8
from __future__ import unicode_literals

from deepnlp import segmenter

## Example 1: Base Model
tokenizer = segmenter.load_model(name = 'zh')

text = "我爱吃北京烤鸭"
segList = tokenizer.seg(text) # python 2/3: function input: unicode, return unicode
text_seg = " ".join(segList)
print (text_seg)
# 我 爱 吃 北京 烤鸭

## Example 2: Entertainment domain, Movie, Teleplay, Actor name, ...
tokenizer = segmenter.load_model(name = 'zh_entertainment')

text = "我刚刚在浙江卫视看了电视剧老九门，觉得陈伟霆很帅"
segList = tokenizer.seg(text)
text_seg = " ".join(segList)

print (text_seg)
# 我 刚刚 在 浙江卫视 看 了 电视剧 老九门 ， 觉得 陈伟霆 很 帅

## Example 3: o2o domain
tokenizer = segmenter.load_model(name = 'zh_o2o')

text = "三里屯哪里有好吃的北京烤鸭"
segList = tokenizer.seg(text)
text_seg = " ".join(segList)

print (text_seg)
# 哪里 有 好吃的 北京烤鸭
