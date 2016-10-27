#!/usr/bin/env Python  
#coding=utf-8

from distutils.core import setup
from setuptools import find_packages

setup(
    name = 'deepnlp',
    version = '0.1.0',
    description = 'Deep Learning NLP Pipeline implemented on Tensorflow',
    author = 'Xichen Ding',
    author_email = 'dingo0927@126.com',
    url = 'https://github.com/rockingdingo/deepnlp',
    license="MIT",
    keywords='Deep Learning,NLP,Tensorflow,deepnlp.org',
    packages=find_packages(),
    package_data={
        'deepnlp': ['segment/data/crf_model', 
            'pos/data/word_to_id', 'pos/data/tag_to_id', 'pos/ckpt/*',
            'ner/data/word_to_id', 'ner/data/tag_to_id', 'ner/ckpt/*',
        ],
    },
    requires=['numpy', 'tensorflow', 'CRFPP',
    ],
)