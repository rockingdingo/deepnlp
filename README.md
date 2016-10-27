About 'deepnlp'
========
Deep Learning NLP Pipeline implemented on Tensorflow purely by python.
Following the 'simplicity' rule, this project aims to provide an easy python version implementation of NLP pipeline based on Tensorflow platform.
It serves the same purpose as Google SyntaxNet but is clearer, easier to install, read and extend(purely in python). 
It also require fewer dependency. (Installing Bazel is painful on my machine...)

* NLP Pipeline Modules:
    * Word Segmentation/Tokenization
    * Part-of-speech (POS)
    * Named-entity-recognition(NER)
    * Planed: Parsing, Automatic Summarization

* Algorithm(Closely following the state-of-Art)
    * Word Segmentation: Linear Chain CRF(conditional-random-field), based on python CRF++ module
    * POS: LSTM/BI-LSTM network, based on Tensorflow
    * NER: LSTM/BI-LSTM/LSTM-CRF network, based on Tensorflow

* Pre-trained Model
    * Chinese: Segmentation, POS, NER
    * English: To be released soon.
    * For your Specific Language, you can easily use the shell to train model with the corpus of your language choice.

Installation
========
* Requirements
    * Tensorflow(>=0.10.0)   LSTM module in Tensorflow change a lot since 0.10.0 compared to previous versions
    * CRF++ (>=0.54)         Download from: https://taku910.github.io/crfpp/

* Pip
```python
    # windows: tensorflow is not supported on windows right now, so is deepnlp
    # linux, run the script:
    pip install deepnlp
```

Tutorial
========
see below section

deepnlp 项目简介
========
deepnlp项目是基于Tensorflow平台的一个python版本的NLP套装, 目的在于将Tensorflow深度学习平台上的模块，结合
最新的一些算法，提供NLP基础模块的支持，并支持其他更加复杂的任务的拓展，如生成式文摘，对话等等。

* NLP 套装模块
    * 分词 Word Segmentation/Tokenization
    * 词性标注 Part-of-speech (POS)
    * 命名实体识别 Named-entity-recognition(NER)
    * 计划中: 句法分析 Parsing, 自动生成式文摘 Automatic Summarization

* 算法实现
    * 分词: 线性链条件随机场 Linear Chain CRF, 基于CRF++包来实现
    * 词性标注: 单向LSTM/ 双向BI-LSTM, 基于Tensorflow实现
    * 命名实体识别: 单向LSTM/ 双向BI-LSTM/ LSTM-CRF 结合网络, 基于Tensorflow实现

* 预训练模型
    * 中文: 基于人民日报语料和微博混合语料: 分词, 词性标注, 实体识别


安装说明
=======
* 需要
    * Tensorflow(>=0.10.0)   Tensorflow 0.10.0 里的LSTM模块和以前版本相比有较大变化，所以尽量以最新的为基准；
    * CRF++ (>=0.54)         可以从 https://taku910.github.io/crfpp/ 下载安装

* Pip 安装
```python
    # windows: tensorflow is not supported on windows right now, so is deepnlp
    # linux, run the script:
    pip install deepnlp
```

Tutorial 教程
========
Set Coding 设置编码
```python
#encoding=utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
print sys.getdefaultencoding()
```

Segmentation 分词模块
```python
import deepnlp.segmenter as segmenter

text = "我爱吃北京烤鸭"
segList = segmenter.seg(text.decode('utf-8')) # python 2: function input: unicode, return unicode
text_seg = " ".join(segList)

print (text.encode('utf-8'))
print (text_seg.encode('utf-8'))
```

POS 词性标注
```python
import deepnlp.segmenter as segmenter
import deepnlp.pos_tagger as pos_tagger

#Segmentation
text = "我爱吃北京烤鸭"
words = segmenter.seg(text.decode('utf-8'))  # chinese characters are using unicode in python 2.7
print (" ".join(words).encode('utf-8'))

#POS Tagging
tagging = pos_tagger.predict(words)
for (w,t) in tagging:
    str = w + "/" + t
    print (str.encode('utf-8'))
```

NER 命名实体识别
```python
import deepnlp.segmenter as segmenter
import deepnlp.ner_tagger as ner_tagger

#Segmentation
text = "我爱吃北京烤鸭"
words = segmenter.seg(text.decode('utf-8'))
print (" ".join(words).encode('utf-8'))

#NER tagging
tagging = ner_tagger.predict(words)
for (w,t) in tagging:
    str = w + "/" + t
    print (str.encode('utf-8'))
```

Pipeline 通过Pipeline调用返回多个值
```python
import deepnlp.pipeline as p

#Segmentation
text = "我爱吃北京烤鸭"
res = p.analyze(text.decode('utf-8'))

print (res[0].encode('utf-8'))
print (res[1].encode('utf-8'))
print (res[2].encode('utf-8'))
```

