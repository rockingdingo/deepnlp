import deepnlp

from deepnlp import segmenter
try:
    deepnlp.download("segment", "zh_finance")
except Exception as e:
    print (e)

deepnlp.register_model("segment", "zh_finance")
deepnlp.download("segment", "zh_finance")
seg_tagger = segmenter.load_model("zh")

from deepnlp import pos_tagger
try:
    deepnlp.download("pos", "zh_finance")
except Exception as e:
    print (e)

deepnlp.register_model("pos", "zh_finance")
deepnlp.download("pos", "zh_finance")
pos_tagger.load_model("zh_finance")

from deepnlp import ner_tagger
try:
    deepnlp.download("ner", "zh_finance")
except Exception as e:
    print (e)

deepnlp.register_model("ner", "zh_finance")
deepnlp.download("ner", "zh_finance")
ner_tagger.load_model("zh_finance")

from deepnlp import nn_parser
try:
    deepnlp.download("parse", "zh_finance")
except Exception as e:
    print (e)

deepnlp.register_model("parse", "zh_finance")
deepnlp.download("parse", "zh_finance")
nn_parser.load_model("zh_finance")

