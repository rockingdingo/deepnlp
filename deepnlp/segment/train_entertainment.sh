# Merge Base Data
cat data/zh/train.txt \
    data/zh_entertainment/weibo_train.txt \
> data/zh_entertainment/train_merge.txt


python data_util.py ${LOCAL_PATH}/data/zh_entertainment/train_merge.txt ${LOCAL_PATH}/data/zh_entertainment/train_word_tag.txt


crf_learn -f 3 -c 4.0 ${LOCAL_PATH}/data/zh_entertainment/template ${LOCAL_PATH}/data/zh_entertainment/train_word_tag.txt ${LOCAL_PATH}/models/zh_entertainment/crf_model


