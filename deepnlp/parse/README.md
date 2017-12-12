Dependency Parsing
==============================
句法依存解析

We implemented the arc-standard parsing system and used 
simple feed-forward neural networks to predict the correct arc label and head of the word.
This python script is a rewrite version of NNDepParser in Stanford CoreNLP, following the work of Chen and Manning.
A Fast and Accurate Dependency Parser using Neural Networks
(http://cs.stanford.edu/people/danqi/papers/emnlp2014.pdf)

Usage
--------------------
使用

####  Prediction
## 1. Input: Words and Pos Tags; POS Tags should be aligned with the dependency tree corpus
```python

from __future__ import unicode_literals # compatible with python3 unicode coding

from deepnlp import nn_parser
parser = nn_parser.load_model(name = 'zh')

#Example 1, Input Words and Tags Both
words = ['它', '熟悉', '一个', '民族', '的', '历史']
tags = ['r', 'v', 'm', 'n', 'u', 'n']

#Parsing
dep_tree = parser.predict(words, tags)
print ("id" + "\t" + "form" + "\t" + "lemma" + "\t" + "pos" + "\t" + "ppos"+ "\t" + "head" + "\t" + "deprel")
print (dep_tree)

```

## 2. Input: Words, Using the default POS Tagger of deepnlp
```python

from __future__ import unicode_literals # compatible with python3 unicode coding

from deepnlp import nn_parser
parser = nn_parser.load_model(name = 'zh')

#Example 2, Input Words and Using the default Pos Tags
words = ['这种', '局面', '的', '形成', '有着', '复杂', '的', '社会', '背景']

#Parsing
dep_tree = parser.predict(words)
print ("id" + "\t" + "form" + "\t" + "lemma" + "\t" + "pos" + "\t" + "ppos"+ "\t" + "head" + "\t" + "deprel")
print (dep_tree)

```

## 3. Fetching Result from namedTuple Transition
```python

num_token = dep_tree.count()
print ("id\tword\tpos\thead\tlabel")
for i in range(num_token):
    cur_id = str(dep_tree.tree[i+1].id)
    cur_form = str(dep_tree.tree[i+1].form)
    cur_pos = str(dep_tree.tree[i+1].pos)
    cur_head = str(dep_tree.tree[i+1].head)
    cur_label = str(dep_tree.tree[i+1].deprel)
    print ("%s\t%s\t%s\t%s\t%s" % (cur_id, cur_form, cur_pos, cur_head, cur_label))

#Result:
#1	它	r	2	SBV
#2	熟悉	v	0	HED
#3	一个	m	4	QUN
#4	民族	n	5	DE
#5	的	u	6	ATT
#6	历史	n	2	VOB

```


Train your model
--------------------
自己训练模型

#### Corpus and Format
See CONLL 2006/2009 data format for details
The training corpus are from below source with headers as follows:
ID FORM LEMMA POS PPOS _ HEAD DEPREL _ _

Example:
```python
1	用	用	p	p	_	4	ADV	_	_
2	先进	先进	a	a	_	3	ATT	_	_
3	典型	典型	n	n	_	1	POB	_	_
4	推动	推动	v	v	_	0	HED	_	_
5	部队	部队	n	n	_	7	ATT	_	_
6	全面	全面	a	a	_	7	ATT	_	_
7	建设	建设	v	v	_	4	VOB	_	_

```

#### Training models
```python
cd ./parse
python parse_model.py  # NN model for arc-standard parsing system 

```


#### Feature Template
NN Dep Parser use the same word, pos tag and deprel embedding features as in paper(Chen and Manning).
This implementation use feature template configuration file to generate features. Sample features
are shown as below:

```python
# Word Embedding Features, total 18
WORD_STACK[0]   # word feature: first element on stack
WORD_STACK[1]
WORD_STACK[2]
WORD_BUFFER[0]  # word feature: first element on buffer
WORD_BUFFER[1]
WORD_BUFFER[2]
WORD_LC[0](STACK[0])   # word feature: left child of first element on stack
WORD_LC[1](STACK[0])
WORD_LC[0](STACK[1])
...

# POS Tag Features, total 18
POS_STACK[0]
POS_STACK[1]
...

# dependency relation features, total 12
DEPREL_LC[0](STACK[0])
DEPREL_LC[1](STACK[0])
DEPREL_LC[0](STACK[1])

```

#### Training Instance
Training instance are generated and saved using pickle file for future use.
They are fisted generated and saved the first time when below functions are called..
```python

# function for generating train/dev data
reader.load_data(data_path=None)

# train_trees.pkl
# dev_trees.pkl
# train_sents.pkl
# dev_sents.pkl

# function for generating train/dev examples
transition_system.generate_examples(sents, trees, label_dict, feature_tpl, instance_path, is_train = True):

# train_examples_X.pkl
# train_examples_Y.pkl
# dev_examples_X.pkl
# dev_examples_Y.pkl
```
