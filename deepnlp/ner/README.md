NER (Named Entity Recognition)
==============================
命名实体识别


Prediction
--------------------
预测

```python

from __future__ import unicode_literals # compatible with python3 unicode

from deepnlp import ner_tagger

tagger = ner_tagger.load_model(name = 'zh_o2o')   # Base LSTM Based Model + zh_o2o dictionary
text = "北京 望京 最好吃 的 小龙虾 在 哪里"
words = text.split(" ")
tagging = tagger.predict(words, tagset = ['city', 'area', 'dish'])
for (w,t) in tagging:
    pair = w + "/" + t
    print (pair)

#Result
#北京/city
#望京/area
#最好吃/nt
#的/nt
#小龙虾/dish
#在/nt
#哪里/nt

```

Train your model
--------------------
自己训练模型

### LSTM-based Sequential Tagging
```python
python ner_model.py en # LSTM model English

python ner_model.py zh # LSTM model Chinese

```

### Bi-LSTM based Sequential Tagging
```python
python ner_model_bilstm.py en # Bi-LSTM model English

python ner_model_bilstm.py zh # Bi-LSTM model Chinese

```

### Bi-LSTM-CRF based Sequential Tagging
```python
python ner_model_bilstm_crf.py en # Bi-LSTM model English

python ner_model_bilstm_crf.py zh # Bi-LSTM model Chinese

```

###NER model Corpus Preparation
#### Prepare corpus the same way as POS
#### Put data files in folder ../deepnlp/ner/data/'your_model_name'
#### Running script
```python
python ner_model.py zh # training Chinese model
```
#### The trained model can be found under folder ../deepnlp/ner/ckpt/'your_model_name'


Domain-specific NER models
--------------------------------------
与训练的领域相关的NER模型

Below pre-trained NER models are provided. You can download the models
easily by calling deepnlp.download() function

### NER Domain models

| Module        | Model            | Note      | Entity      | Status               |
| ------------- |:-----------------|:----------|:------------|:---------------------|
| NER           | zh               | 中文      |  city(城市),district(区域),area(商圈)   | Release  |
| NER           | zh_entertainment | 中文,文娱  |  actor(演员),role_name(角色名),teleplay(影视剧),teleplay_tag(影视剧标签) | Release  |
| NER           | zh_o2o | 中文,O2O  |  dish(菜品名),shop(店名),category(菜品类目名) | Release  |
| NER           | zh_finance       | 中文,财经  |  To Do | Contribution Welcome  |
| NER           | other domain are welcome   |   |  To Do | Contribution Welcome  |

### Download models

```python
import deepnlp
deepnlp.download(module='ner', name='zh_entertainment')  # download the entertainment model

```

Domain-specific NER models
--------------------------------------
预训练细分领域模型

###
不同领域的实体识别或槽位解析的效果非常依赖于语料和相关词典, 目前很多领域工作都需要在通用模型基础上重新造轮子。
deepnlp 希望本着开放和共享的理念,在保护版权基础上, 提供基于各个领域的语料训练的基础模型和领域词典来满足常见需求。
也欢迎大家尽一份力, 贡献一些自己积累的素材。
目前提供了包括下列领域的深度Tensorflow模型和词典, 未添加的领域欢迎Contributor来贡献:
娱乐(entertainment): 常见电影,电视剧,娱乐词典;
泛行业(O2O): 菜品, 商铺, 菜系等;



User Dictionary and UDF disambiguation
----------------------------------------
用户实体词典和UDF的简单消歧

### User Dictionary Format
Suppose you have prepared a entity_tags dictionary file, 'entity_tags.dic', 
which has the format as "word \t tag" in a line

```python
胡歌	actor
猎场	teleplay
...

```

See ./test/test_ner_dict_udf.py for details

```python
## Load User Dict
from deepnlp import ner_tagger
tagger = ner_tagger.load_model(name = 'zh') 
dict_path = "./your_dict_path/entity_tags.dic"
tagger.load_user_dict(dict_path)



## Simple UDF for disambiguation

from deepnlp.ner_tagger import udf_disambiguation_cooccur
from deepnlp.ner_tagger import udf_default

word = "琅琊榜"
tags = ['list_name', 'teleplay']
context = ["今天", "我", "看", "了"]
tag_feat_dict = {}
# Most Freq Word Feature of two tags
tag_feat_dict['list_name'] = ['听', '专辑', '音乐']
tag_feat_dict['teleplay'] = ['看', '电视', '影视']
# Disambuguiation Prob
tag, prob = udf_disambiguation_cooccur(word, tags, context, tag_feat_dict)
print ("DEBUG: NER tagger zh_entertainment with user defined function for disambuguiation")
print ("Word:%s, Tag:%s, Prob:%f" % (word, tag, prob))

# Combine the results and load the udfs
tagging = tagger._predict_ner_tags_dict(words, merge = True, udfs = [udf_disambiguation_cooccur])
for (w,t) in tagging:
    pair = w + "/" + t
    print (pair)

```

