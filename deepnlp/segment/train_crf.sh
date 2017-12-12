#!/bin/bash

LOCAL_PATH="$(cd `dirname $0`; pwd)"
echo 'Prepare Training Data File: train_word_tag.txt'
python data_util.py ${LOCAL_PATH}/data/zh/train.txt ${LOCAL_PATH}/data/zh/train_word_tag.txt
python data_util.py ${LOCAL_PATH}/data/zh/test.txt ${LOCAL_PATH}/data/zh/test_word_tag.txt

# Train Model Using CRF++ command, template: template file; train_word_tag.txt: data file; crf_model: model name;
crf_learn -f 3 -c 4.0 ${LOCAL_PATH}/data/zh/template ${LOCAL_PATH}/data/zh/train_word_tag.txt ${LOCAL_PATH}/models/zh/crf_model

# See how to predict, check the segmenter.py file under /deepnlp/segmenter.py
