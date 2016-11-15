#!/usr/bin/env Python  
#coding=utf-8

from distutils.core import setup
from setuptools import find_packages

setup(
    name = 'deepnlp',
    version = '0.1.5',
    description = 'Deep Learning NLP Pipeline implemented on Tensorflow',
    author = 'Xichen Ding',
    author_email = 'dingo0927@126.com',
    url = 'https://github.com/rockingdingo/deepnlp',
    license="MIT",
    keywords='DeepLearning,NLP,Tensorflow,deepnlp.org',
    packages=find_packages(),
    package_data={
        'deepnlp': ['segment/data/crf_model', 
            'segment/data/template',
            'segment/train_crf.sh',
            'pos/trainPOSModel.sh',
            'pos/data/zh/word_to_id', 'pos/data/zh/tag_to_id', 'pos/ckpt/zh/*',
            'pos/data/en/word_to_id', 'pos/data/en/tag_to_id', 'pos/ckpt/en/*',
            'ner/data/zh/word_to_id', 'ner/data/zh/tag_to_id', 'ner/ckpt/zh/*',
        ],
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