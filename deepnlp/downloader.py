#!/usr/bin/python
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals # compatible with python3 unicode coding

import sys, os
import urllib
if (sys.version_info>(3,0)): from urllib.request import urlretrieve 
else : from urllib import urlretrieve

pkg_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(pkg_path)

def Schedule(a,b,c):
    '''''
    a: Data already downloaded
    b: Size of Data block
    c: Size of remote file
   '''
    per = 100.0 * a * b / c
    per = min(100.0, per)
    if(int(round(per, 2)) % 10 == 0):
        print ('Downloading %.2f%%' % per)

github_repo = "https://github.com/rockingdingo/deepnlp/raw/master/deepnlp"
folder = os.path.dirname(os.path.abspath(__file__))

model_ner = [
(github_repo + "/ner/data/zh/" + "word_to_id",  folder + "/ner/data/zh/" + "word_to_id"),   # Chinese model
(github_repo + "/ner/data/zh/" + "tag_to_id",  folder + "/ner/data/zh/" + "tag_to_id"),
(github_repo + "/ner/ckpt/zh/" + "checkpoint", folder + "/ner/ckpt/zh/" + "checkpoint"),
(github_repo + "/ner/ckpt/zh/" + "ner.ckpt.data-00000-of-00001", folder + "/ner/ckpt/zh/" + "ner.ckpt.data-00000-of-00001"),
(github_repo + "/ner/ckpt/zh/" + "ner.ckpt.index", folder + "/ner/ckpt/zh/" + "ner.ckpt.index"),
(github_repo + "/ner/ckpt/zh/" + "ner.ckpt.meta", folder + "/ner/ckpt/zh/" + "ner.ckpt.meta"),
]

model_pos = [
(github_repo + "/pos/data/zh/" + "word_to_id",  folder + "/pos/data/zh/" + "word_to_id"),   # Chinese model
(github_repo + "/pos/data/zh/" + "tag_to_id",  folder + "/pos/data/zh/" + "tag_to_id"),
(github_repo + "/pos/ckpt/zh/" + "checkpoint",  folder + "/pos/ckpt/zh/" + "checkpoint"),   
(github_repo + "/pos/ckpt/zh/" + "pos.ckpt.data-00000-of-00001", folder + "/pos/ckpt/zh/" + "pos.ckpt.data-00000-of-00001"),  
(github_repo + "/pos/ckpt/zh/" + "pos.ckpt.index", folder + "/pos/ckpt/zh/" + "pos.ckpt.index"),  
(github_repo + "/pos/ckpt/zh/" + "pos.ckpt.meta", folder + "/pos/ckpt/zh/" + "pos.ckpt.meta"),
(github_repo + "/pos/data/en/" + "word_to_id",  folder + "/pos/data/en/" + "word_to_id"),   # English model
(github_repo + "/pos/data/en/" + "tag_to_id",  folder + "/pos/data/en/" + "tag_to_id"),
(github_repo + "/pos/ckpt/en/" + "checkpoint",  folder + "/pos/ckpt/en/" + "checkpoint"),
(github_repo + "/pos/ckpt/en/" + "pos.ckpt.data-00000-of-00001", folder + "/pos/ckpt/en/" + "pos.ckpt.data-00000-of-00001"),  
(github_repo + "/pos/ckpt/en/" + "pos.ckpt.index", folder + "/pos/ckpt/en/" + "pos.ckpt.index"),  
(github_repo + "/pos/ckpt/en/" + "pos.ckpt.meta", folder + "/pos/ckpt/en/" + "pos.ckpt.meta"),
]

model_segment = [
(github_repo + "/segment/data/" + "crf_model", folder + "/segment/data/" + "crf_model"),
(github_repo + "/segment/data/" + "template", folder + "/segment/data/" + "template"),
]

model_textsum = [
(github_repo + "/textsum/ckpt/" + "checkpoint", folder + "/textsum/ckpt/" + "checkpoint"),
(github_repo + "/textsum/ckpt/" + "headline_large.ckpt-48000.data-00000-of-00001.tar.gz00", folder + "/textsum/ckpt/" + "headline_large.ckpt-48000.data-00000-of-00001.tar.gz00"),
(github_repo + "/textsum/ckpt/" + "headline_large.ckpt-48000.data-00000-of-00001.tar.gz01", folder + "/textsum/ckpt/" + "headline_large.ckpt-48000.data-00000-of-00001.tar.gz01"),
(github_repo + "/textsum/ckpt/" + "headline_large.ckpt-48000.data-00000-of-00001.tar.gz02", folder + "/textsum/ckpt/" + "headline_large.ckpt-48000.data-00000-of-00001.tar.gz02"),
(github_repo + "/textsum/ckpt/" + "headline_large.ckpt-48000.index", folder + "/textsum/ckpt/" + "headline_large.ckpt-48000.index"),
(github_repo + "/textsum/news/" + "vocab", folder + "/textsum/news/" + "vocab"),
(github_repo + "/textsum/news/train/" + "content-train-sample.txt", folder + "/textsum/news/train/" + "content-train-sample.txt"),
(github_repo + "/textsum/news/train/" + "title-train-sample.txt", folder + "/textsum/news/train/" + "title-train-sample.txt"),
(github_repo + "/textsum/news/test/" + "content-test.txt", folder + "/textsum/news/test/" + "content-test.txt"),
(github_repo + "/textsum/news/test/" + "title-test.txt", folder + "/textsum/news/test/" + "title-test.txt"),
(github_repo + "/textsum/news/test/" + "summary.txt", folder + "/textsum/news/test/" + "summary.txt"),
]

def download_model(models):
    for url, localfile in models:
        if os.path.exists(localfile):
            print ("Local file %s exists..." % localfile)
        else:
            print ("Downloading file %s" % url)
            dir= os.path.dirname(localfile)  # create dir for local file if not exist
            if not os.path.exists(dir):
                os.makedirs(dir)
            urlretrieve(url, localfile, Schedule)

def download(module = None):
    print ("Starting download models from remote github repo...")
    if module:
        if (module.lower() == "segment"):
            print ("Downloading Segment module...")
            download_model(model_segment)
        elif (module.lower() == "pos"):
            print ("Downloading POS module...")
            download_model(model_pos)
        elif (module.lower() == "ner"):
            print ("Downloading NER module...")
            download_model(model_ner)
        elif (module.lower() == "textsum"):
            print ("Downloading Textsum module...")
            download_model(model_textsum)
        else:
            print ("module not found...")
    
    else:
        # default download all the require models
        print ("Downloading Segment, POS, NER, Textsum module...")
        download_model(model_segment)
        download_model(model_pos)
        download_model(model_ner)
        download_model(model_textsum)

def test():
    download('pos')

if __name__ == '__main__':
    test()

