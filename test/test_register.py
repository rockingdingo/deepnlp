#!/usr/bin/python
# -*- coding:utf-8 -*-

import deepnlp

from deepnlp import segmenter
try:
    deepnlp.download("segment", "zh_finance")
except Exception as e:
    print (e)

deepnlp.register_model("segment", "zh_finance")
deepnlp.download("segment", "zh_finance")
try:
    seg_tagger = segmenter.load_model("zh_finance")
except Exception as e:
    print (e)

from deepnlp import pos_tagger
try:
    deepnlp.download("pos", "zh_finance")
except Exception as e:
    print (e)

deepnlp.register_model("pos", "zh_finance")
deepnlp.download("pos", "zh_finance")
try:
    pos_tagger.load_model("zh_finance")
except Exception as e:
    print (e)

from deepnlp import ner_tagger
try:
    deepnlp.download("ner", "zh_finance")
except Exception as e:
    print (e)

deepnlp.register_model("ner", "zh_finance")
deepnlp.download("ner", "zh_finance")
try:
    ner_tagger.load_model("zh_finance")
except Exception as e:
    print (e)

from deepnlp import nn_parser
try:
    deepnlp.download("parse", "zh_finance")
except Exception as e:
    print (e)

deepnlp.register_model("parse", "zh_finance")
deepnlp.download("parse", "zh_finance")
try:
    nn_parser.load_model("zh_finance")
except Exception as e:
    print (e)
