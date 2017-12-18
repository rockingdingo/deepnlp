#!/bin/bash
LOCAL_PATH="/Users/dingxichen/Desktop/data/python/pypi/deepnlp_py3/deepnlp"
## POS Zh model
python ${LOCAL_PATH}/pos/pos_model.py zh
## POS En model
python ${LOCAL_PATH}/pos/pos_model.py en
