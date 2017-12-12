#!/usr/bin/python
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals # compatible with python3 unicode coding

import sys, os
import urllib
if (sys.version_info>(3,0)): from urllib.request import urlretrieve, urlopen
else : from urllib import urlretrieve, urlopen

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
    prev_break_point = -1
    if(int(per) % 10 == 0):
        cur_break_point = int(per)
        if (cur_break_point != prev_break_point):
            print ('Downloading %.2f%%' % per)
            prev_break_point = cur_break_point

# Download from below two sources
github_repo = "https://github.com/rockingdingo/deepnlp/raw/master/deepnlp"
deepnlp_repo = "http://deepnlp.org/downloads/deepnlp/models"

folder = os.path.dirname(os.path.abspath(__file__))

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

segment_model_list = ['zh', 'zh_o2o', 'zh_entertainment']
ner_model_list = ['zh', 'zh_o2o', 'zh_entertainment']
pos_model_list = ['zh', 'en']
parse_model_list = ['zh']

def get_model_ner(model_name_list):
    """ Args: model relative path
        Return: relative path of all related files
    """
    data_path = "ner/data/"
    ckpt_path = "ner/ckpt/"
    data_files = ["word_to_id", "tag_to_id"]
    ckpt_files = ["checkpoint", "ner.ckpt.data-00000-of-00001", "ner.ckpt.index", "ner.ckpt.meta"]
    model_ner = []
    for model_name in model_name_list:
        # Append data files
        for data_file in data_files:
            relative_data_file_path = data_path + model_name + "/" + data_file
            model_ner.append((relative_data_file_path, relative_data_file_path))
        # Append Ckpt files
        for ckpt_file in ckpt_files:
            relative_ckpt_file_path = ckpt_path + model_name + "/" + ckpt_file
            model_ner.append((relative_ckpt_file_path, relative_ckpt_file_path))
    return model_ner

def get_model_pos(model_name_list):
    """ Args: model relative path
        Return: relative path of all related files
    """
    data_path = "pos/data/"
    ckpt_path = "pos/ckpt/"
    data_files = ["word_to_id", "tag_to_id"]
    ckpt_files = ["checkpoint", "pos.ckpt.data-00000-of-00001", "pos.ckpt.index", "pos.ckpt.meta"]
    model_pos = []
    for model_name in model_name_list:
        # Append data files
        for data_file in data_files:
            relative_data_file_path = data_path + model_name + "/" + data_file
            model_pos.append((relative_data_file_path, relative_data_file_path))
        # Append Ckpt files
        for ckpt_file in ckpt_files:
            relative_ckpt_file_path = ckpt_path + model_name + "/" + ckpt_file
            model_pos.append((relative_ckpt_file_path, relative_ckpt_file_path))
    return model_pos

def get_model_segment(model_name_list):
    model_path = "segment/models/"
    data_path = "segment/data/"
    model_files = ["crf_model"]
    data_files = ["template"]
    model_segment = []
    for model_name in model_name_list:
        # Append data files
        for data_file in data_files:
            relative_data_file_path = data_path + model_name + "/" + data_file
            model_segment.append((relative_data_file_path, relative_data_file_path))
        # Append Ckpt files
        for model_file in model_files:
            relative_model_file_path = model_path + model_name + "/" + model_file
            model_segment.append((relative_model_file_path, relative_model_file_path))
    return model_segment

def get_model_parse(model_name_list):
    model_parse = []
    # ckpt path and files
    ckpt_path = "parse/ckpt/"
    ckpt_files = ["checkpoint", "parse.ckpt.data-00000-of-00001", "parse.ckpt.index", "parse.ckpt.meta"]
    # data path and files
    data_path = "parse/data/"
    data_files = ["parse.template", "pos_dict", "vocab_dict", "label_dict"]
    # combine
    model_parse = []
    for model_name in model_name_list:
        # Append data files
        for data_file in data_files:
            relative_data_file_path = data_path + model_name + "/" + data_file
            model_parse.append((relative_data_file_path, relative_data_file_path))
        # Append Ckpt files
        for ckpt_file in ckpt_files:
            relative_model_file_path = ckpt_path + model_name + "/" + ckpt_file
            model_parse.append((relative_model_file_path, relative_model_file_path))
    return model_parse

def download_model(models):
    """ Args:
            models: list of tuples [relative_repo_path, relative_local_file_path]
            repo: URL of repo, github or www.deepnlp.org
            folder: local folder to save the file
    """
    for rel_repo_path, rel_local_file in models:
        localfile = os.path.join(folder, rel_local_file)
        if os.path.exists(localfile):
            print ("NOTICE: Local file %s exists..." % localfile)
        else:   # local model doesn't exist
            # Create local Directory
            local_dir= os.path.dirname(localfile)  # create dir for local file if not exist
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)
            # First try if deepnlp repo resource exist
            url_deepnlp = deepnlp_repo + "/" + rel_repo_path   # .../models + "/pos/..."
            url_git = github_repo + "/" + rel_repo_path
            #print ("DEBUG: Repo %s" % url_deepnlp)
            #print ("DEBUG: Repo %s" % url_git)
            # First try if deepnlp repo resource exist
            deepnlp_ret = None
            try:
                deepnlp_ret = urlopen(url_deepnlp, data = None, timeout = 3)
            except Exception as e:
                deepnlp_ret = None
                #print ("DEBUG: Downloading from deepnlp met error:")
                #print (e)
            # Check if deepnlp repo return True
            deepnlp_ret_suc = False
            if deepnlp_ret is not None:
                if deepnlp_ret.code == 200:
                    deepnlp_ret_suc = True
            if (deepnlp_ret_suc):   # Download from deepnlp repo, return Success
                print ("NOTICE: Downloading models from repo %s to local %s" % (url_deepnlp, localfile))
                try:
                    urlretrieve(url_deepnlp, localfile, Schedule)
                except:
                    print ("Debug: Failed to download models from repo %s, met connection error" % url_deepnlp)
                    continue
            else:   # Donwload from github
                print ("NOTICE: Downloading models from repo %s to Local %s" % (url_git, localfile))
                try:
                    urlretrieve(url_git, localfile, Schedule)
                except:
                    print ("Debug: Failed to download models from Repo %s met error" % url_git)
                    continue

def download(module = None, name = None):
    """ Args: module: e.g. POS, NER, Parse, ....
            name: model name , e.g. zh, zh_o2o, zh_entertainment
    """
    if module is not None:
        if (module.lower() == "segment"):
            if name is not None:    # Download Specific Model
                cur_model_segment = get_model_segment([name])
                download_model(cur_model_segment)
            else:
                print ("NOTICE: Downloading Segment module all ...")
                model_segment = get_model_segment(segment_model_list)
                download_model(model_segment)
        elif (module.lower() == "pos"):
            if name is not None:
                print ("NOTICE: Downloading POS module %s" % name)
                cur_model_pos = get_model_pos([name])
                download_model(cur_model_pos)
            else:
                print ("NOTICE: Downloading POS module All ...")
                model_pos = get_model_pos(pos_model_list)
                download_model(model_pos)
        elif (module.lower() == "ner"):
            if name is not None:
                print ("NOTICE: Downloading NER module %s" % name)
                cur_model_ner = get_model_ner([name])
                download_model(cur_model_ner)
            else:
                print ("NOTICE: Downloading NER module All ...")
                model_ner = get_model_ner(ner_model_list)
                download_model(model_ner)
        elif (module.lower() == "parse"):
            if name is not None:
                print ("NOTICE: Downloading Parse module %s" % name)
                cur_model_parse = get_model_parse([name])
                download_model(cur_model_parse)
            else:
                print ("NOTICE: Downloading Parse module All ...")
                model_parse = get_model_parse(parse_model_list)
                download_model(model_parse)
        elif (module.lower() == "textsum"):
            print ("NOTICE: Downloading Textsum module...")
            download_model(model_textsum)
        else:
            print ("NOTICE: module not found...")
    else:
        # default download all the require models
        print ("NOTICE: Downloading Segment, POS, NER, Textsum module...")
        model_segment = get_model_segment(segment_model_list)
        model_pos = get_model_pos(pos_model_list)
        model_ner = get_model_ner(ner_model_list)
        model_parse = get_model_parse(parse_model_list)
        download_model(model_segment)
        download_model(model_pos)
        download_model(model_ner)
        download_model(model_parse)
        #download_model(model_textsum)

def test():
    download('pos')
    download('pos', 'en')

if __name__ == '__main__':
    test()
