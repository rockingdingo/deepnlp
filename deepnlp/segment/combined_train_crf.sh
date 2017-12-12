#!/bin/bash

LOCAL_PATH="$(cd `dirname $0`; pwd)"
echo 'Prepare Training Data File: train_word_tag.txt'

# Merge Base Data
cat data/zh/train.txt \
    data/zh_o2o/o2o_train.txt \
> data/zh_o2o/train_merge.txt

cat data/zh/test.txt \
    data/zh_o2o/o2o_test.txt \
> data/zh_o2o/test_merge.txt

python data_util.py ${LOCAL_PATH}/data/zh_o2o/train_merge.txt ${LOCAL_PATH}/data/zh_o2o/train_word_tag.txt
python data_util.py ${LOCAL_PATH}/data/zh_o2o/test_merge.txt ${LOCAL_PATH}/data/zh_o2o/test_word_tag.txt

# Train Model Using CRF++ command, template: template file; train_word_tag.txt: data file; crf_model: model name;
crf_learn -f 3 -c 4.0 ${LOCAL_PATH}/data/zh_o2o/template ${LOCAL_PATH}/data/zh_o2o/train_word_tag.txt ${LOCAL_PATH}/models/zh_o2o/crf_model

# See how to predict, check the segmenter.py file under /deepnlp/segmenter.py
