#!/bin/bash
LOCAL_PATH="/Users/dingxichen/Desktop/data/python/pypi/deepnlp_py3/deepnlp"
## POS Zh model
python ${LOCAL_PATH}/ner/ner_model.py zh
## POS En model
python ${LOCAL_PATH}/ner/ner_model.py zh_o2o

