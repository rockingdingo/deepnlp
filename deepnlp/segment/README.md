Segmentation 
==================
分词模块

Installation
---------------
安装说明

* Requirements
    * CRF++ (>=0.54)

```Bash
# Download CRF++-0.58.tar.gz from: https://taku910.github.io/crfpp/
tar xzvf CRF++-0.58.tar.gz
cd CRF++-0.58
./configure
make && sudo make install

#install CRFPP python api
cd python
python setup.py build
python setup.py install
ln -s /usr/local/lib/libcrfpp.so.0 /usr/lib/

# Potential installation Errors: 
# ImportError: libcrfpp.so.0: cannot open shared object file: No such file or directory
# Fix: Remember to run below command to link
ln /usr/local/lib/libcrfpp.so.0 to /usr/lib/
```

Usage
--------------------
使用

```python
from __future__ import unicode_literals

from deepnlp import segmenter

## Example 1: Base Model
tokenizer = segmenter.load_model(name = 'zh')

text = "我爱吃北京烤鸭"
segList = tokenizer.seg(text) # python 2/3: function input: unicode, return unicode
text_seg = " ".join(segList)
print (text_seg)

```

Train your model
--------------------
自己训练模型

###Segment model
Install CRF++ 0.58
Follow the instructions
https://taku910.github.io/crfpp/#download

#### Folder Structure
```Bash
/deepnlp
./segment
..data_util.py
..train_crf.sh
../data
.../zh/template
.../zh/train.txt
.../zh/train_word_tag.txt
.../{your_domain}/...
../models
.../zh/crf_model
.../zh_o2o/crf_model
.../zh_entertainment/crf_model
.../{your_domain}/crf_model

```

#### Prepare corpus
Split your data into train.txt and test.txt with format of one sentence per each line: "word1 word2 ...".
Put train.txt and test.txt under folder ../deepnlp/segment/data
Run data_util.py to convert data file to word_tag format and get train_word_tag.txt;
For Chinese, we are using 4 tags representing: 'B' Begnning , 'M' Middle, 'E' End and 'S' Single Char
```shell
我 'S'
喜 'B'
欢 'E'
...
```

```python
python data_util.py ./data/zh/train.txt ./data/zh/train_word_tag.txt
```

#### Define template file needed by CRF++
Sample Template file is included in the package
You can specift the unigram and bigram feature template needed by CRF++

#### Train model using CRF++ module
```shell
# Train Model Using CRF++ command
crf_learn -f 3 -c 4.0 ${LOCAL_PATH}/data/zh/template ${LOCAL_PATH}/data/zh/train_word_tag.txt ${LOCAL_PATH}/models/zh/crf_model
```

Domain Specific Segmentation
----------------------------
打造专有细分领域的分词模型

We all face the problem that general-purpose segmentation tools will not
satisfy our domain-specific needs. That's why we want to build a common-interface
so each of us can have access to the domain-specific models trained by expert in the domain with
labelled corpus.

Since the labelled corpus are private-owned intellectual property, we have the vision to only share the pre-trained model and contribute
to the community, so that other user don't have to build the wheel repetitively and also can have get access to high-quality segmentation tools.

Below are the roadmap of the domains that we want to include.
Please feel free to make contact and contribute if you have domain-specific trained model and want to make an contribution...

### Segmentation Domain models

| Module        | Model            | Note      | Status               |
| ------------- |:-----------------|:----------|:---------------------|
| Segment       | zh               | 中文       | Release  |
| Segment       | zh_entertainment | 中文,文娱   | Release  |
| Segment       | zh_finance       | 中文,财经   | Contribution Welcome  |
| Segment       | ... | 其他 |Contribution Welcome... |


----------------------------
常见的中文分词工具无法满足平时的特定领域工作的需求，每个领域的重复造轮子的工作着实浪费时间。
我们期待提供一个公共load模型的接口，希望能够贡献一些预训练好的模型, 打破语料割裂的边界, 避免重复性造轮子。

我们计划提供下列领域的预训练模型, 欢迎大家给社区贡献一份力量。有任何领域专有模型和语料欢迎联系。


### Download models

```python
import deepnlp
deepnlp.download(module='segment', name='zh_entertainment')  # download the entertainment model

```

#### Usage
```python
from __future__ import unicode_literals

from deepnlp import segmenter

## Download Domain Specific Model
deepnlp.download(module='segment', name='zh_o2o')             # lastest master branch on github
deepnlp.download(module='segment', name='zh_entertainment')   # lastest master branch on github

## Example 2: Entertainment domain: Movie, Teleplay, Actor name, ...
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

```





