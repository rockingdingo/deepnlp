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
...template
...train.txt
...train_word_tag.txt
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
python data_util.py train.txt train_word_tag.txt
```

#### Define template file needed by CRF++
Sample Template file is included in the package
You can specift the unigram and bigram feature template needed by CRF++

#### Train model using CRF++ module
```shell
# Train Model Using CRF++ command
crf_learn -f 3 -c 4.0 ${LOCAL_PATH}/data/template ${LOCAL_PATH}/data/train_word_tag.txt crf_model
```

