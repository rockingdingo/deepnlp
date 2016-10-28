deepnlp
========
Deep Learning NLP Pipeline implemented on Tensorflow purely by python.
Following the 'simplicity' rule, this project aims to provide an easy python version implementation of NLP pipeline based on Tensorflow platform.
It serves the same purpose as Google SyntaxNet but is easier to install, read and extend(purely in python), which also require fewer dependency. 
(Installing Bazel on my machine is painful...)

##Brief Introduction
* [Modules](#Modules)
* [Installation](#Installation)
* [Tutorial](#Tutorial)
    * [Segmentation 分词模块](#Segmentation 分词模块)
    * [POS 词性标注](#POS 词性标注)
    * [NER 命名实体识别](#NER 命名实体识别)
    * [Pipeline 调用](#Pipeline 调用)
    * [Train your model 自己训练模型](#Train your model 自己训练模型)
* [deepnlp中文简介](#deepnlp中文简介)
* [安装说明](#安装说明)
* [Reference](#Reference)

Modules
========
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

* Download source: https://pypi.python.org/pypi/deepnlp/0.1.1
```python
    # linux, run the script:
    tar zxvf deepnlp-0.1.1.tar.gz
    cd deepnlp-0.1.1
    python setup.py install
```

Tutorial
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
-----
```python
import deepnlp.segmenter as segmenter

text = "我爱吃北京烤鸭"
segList = segmenter.seg(text.decode('utf-8')) # python 2: function input: unicode, return unicode
text_seg = " ".join(segList)

print (text.encode('utf-8'))
print (text_seg.encode('utf-8'))
```

POS 词性标注
-----
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
-----
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

Pipeline 调用
-----
```python
import deepnlp.pipeline as p

#Segmentation
text = "我爱吃北京烤鸭"
res = p.analyze(text.decode('utf-8'))

print (res[0].encode('utf-8'))
print (res[1].encode('utf-8'))
print (res[2].encode('utf-8'))
```

Train your model 自己训练模型
-----

###Segment model
Install CRF++ 0.58
Following the instructions
https://taku910.github.io/crfpp/#download

#### Folder Structure
/deepnlp
./segment
..data_util.py
..train_crf.sh
../data
...template
...train.txt
...train_word_tag.txt

* 1. Prepare corpus
Split your data into train.txt and test.txt with format of one sentence per each line: "word1 word2 ...".
Put train.txt and test.txt under folder ../deepnlp/segment/data
Run data_util.py to convert data file to word_tag format and get train_word_tag.txt;
For Chinese, we are using 4 tags representing: 'B' Begnning , 'M' Middle, 'E' End and 'S' Single Char
我 'S'
喜 'B'
欢 'E'
...

"
```python
python data_util.py train.txt train_word_tag.txt
```

* 2. Define template file needed by CRF++
Sample Template file is included in the package
You can specift the unigram and bigram feature template needed by CRF++

* 3. Train model using CRF++ module

```shell
# Train Model Using CRF++ command
crf_learn -f 3 -c 4.0 ${LOCAL_PATH}/data/template ${LOCAL_PATH}/data/train_word_tag.txt crf_model
```

###POS model
#### Folder Structure
/deepnlp
./pos
..pos_model.py
..reader.py
../data
...train.txt
...dev.txt
...test.txt
../ckpt

* 1. Prepare corpus
First, prepare your corpus and split into 3 files: 'train.txt', 'dev.txt', 'test.txt'.
Each line in the file represents one annotated sentence, in this format: "word1/tag1 word2/tag2 ...", separated by white space.
For example,
```python
#train.txt
#English:
POS/NN tagging/NN is/VBZ now/RB done/VBN in/IN the/DT context/NN of/IN computational/JJ linguistics/NNS ./.

#Chinese:
充满/v  希望/n  的/u  新/a  世纪/n  ——/w  一九九八年/t  新年/t  讲话/n  （/w  附/v  图片/n  １/m  张/q  ）/w  
```

* 2. Specifying data_path
So model can find training data files. Download the source of package and put all three corpus files in the folder ../deepnlp/pos/data
you can change data_path setting in reader.py and pos_model.py

* 3. Running script
```python
python pos_model.py
```
* 4. Trained model can be found under folder ../deepnlp/pos/ckpt

###NER model
* 1. Prepare corpus the same way as POS
* 2. Put data files in folder ../deepnlp/ner/data
* 3. Running script
```python
python ner_model.py
```
* 4. The trained model can be found under folder ../deepnlp/ner/ckpt


deepnlp中文项目简介
========
deepnlp项目是基于Tensorflow平台的一个python版本的NLP套装, 目的在于将Tensorflow深度学习平台上的模块，结合
最新的一些算法，提供NLP基础模块的支持，并支持其他更加复杂的任务的拓展，如生成式文摘等等。

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

* 从源码安装, 下载deepnlp-0.1.1.tar.gz文件: https://pypi.python.org/pypi/deepnlp
```python
    # linux, run the script:
    tar zxvf deepnlp-0.1.1.tar.gz
    cd deepnlp-0.1.1
    python setup.py install
```

Reference
=======
CRF++ package: https://taku910.github.io/crfpp/#download
Tensorflow: https://www.tensorflow.org/
Word Segmentation Blog:
http://www.52nlp.cn/%E4%B8%AD%E6%96%87%E5%88%86%E8%AF%8D%E5%85%A5%E9%97%A8%E4%B9%8B%E5%AD%97%E6%A0%87%E6%B3%A8%E6%B3%954

