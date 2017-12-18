#!/usr/bin/env Python  
#coding=utf-8

from distutils.core import setup
from setuptools import find_packages

setup(
    name = 'deepnlp',
    version = '0.1.7',
    description = 'Deep Learning NLP Pipeline implemented on Tensorflow',
    author = 'Xichen Ding',
    author_email = 'dingo0927@126.com',
    url = 'https://github.com/rockingdingo/deepnlp',
    license="MIT",
    keywords='DeepLearning NLP Tensorflow deepnlp.org',
    packages=find_packages(),
    package_data={
        'deepnlp': [
            'segment/data/zh/template',
            'segment/data/zh/train_crf.sh',        
            'segment/models/zh/crf_model',
            'pos/trainPOSModel.sh',
            'pos/data/zh/word_to_id', 
            'pos/data/zh/tag_to_id', 
            'pos/ckpt/zh/*',
            'ner/data/zh/word_to_id', 
            'ner/data/zh/tag_to_id',             
            'ner/dict/zh/entity_tags.dic.pkl',
            'ner/data/zh_o2o/word_to_id', 
            'ner/data/zh_o2o/tag_to_id',             
            'ner/dict/zh_o2o/entity_tags.dic.pkl',            
            'textrank/docs.txt',
            'parse/data/zh/vocab_dict',
            'parse/data/zh/label_dict',
            'parse/data/zh/pos_dict',
            'parse/data/zh/parse.template',                            
            # 'ner/ckpt/zh/*',
            # 'parse/ckpt/zh/*'
            # 'ner/ckpt/zh_o2o/*',            
        ],
        'test': [
            './*',
        ]
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    requires=['numpy', 'tensorflow', 'CRFPP',
    ],
)
