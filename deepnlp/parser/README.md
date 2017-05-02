Parsing Dependency(WIP)
==============================
句法依存解析
s
Train your model
--------------------
自己训练模型

We implemented the arc-standard parsing system and used 
simple feed-forward neural networks to predict the correct arc label and head of the word.
This python script is rewrite of NNDepParser in Stanford CoreNLP, following the work of Chen and Manning.
A Fast and Accurate Dependency Parser using Neural Networks
(http://cs.stanford.edu/people/danqi/papers/emnlp2014.pdf)

#### Corpus and Format
See CONLL 2006/2009 data format for details
The training corpus are from below source with headers as follows:
ID FORM LEMMA POS PPOS _ HEAD DEPREL _ _

Example:
```python
1	上海	上海	NR	NR	_	2	d-LocPhrase	_	_
2	浦东	浦东	NR	NR	_	6	d-genetive	_	_
3	开发	开发	NN	NN	_	6	s-coordinate	_	_
4	与	与	CC	CC	_	6	aux-depend	_	_
5	法制	法制	NN	NN	_	6	d-domain	_	_
6	建设	建设	NN	NN	_	7	experiencer	_	_
7	同步	同步	VV	VV	_	0	ROOT	_	_
```

#### Training models
```python
cd ./parser
python parse_model.py  # NN model for arc-standard parsing system 

```

#### Prediction
```python
python predict.py

# Output


```
