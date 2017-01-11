#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Testing textrank Algorithm
Use deepnlp.org web API for word segmentation
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys, os
import codecs
import requests
import json
from urllib import quote

# Read document and segment words
docs = []
folder_path = os.path.dirname(os.path.abspath(__file__))
#folder_path = "D:\\Python\\tensorflow\\tutorial\\textrank"
file_path = os.path.join(folder_path, 'docs.txt')
f = codecs.open(file_path, encoding='utf-8')
for line in f:
    lines = line.split("。")
    docs.extend(lines)
f.close()

# calling segmentation api
from deepnlp import api_service
# login = api_service.init()          # registration, if failed, load default with limited access
login = {}
conn = api_service.connect(login)   # save the connection with login cookies

base_url = 'www.deepnlp.org'
lang = 'zh'

docSplit = []
for i in range(len(docs)):
    line = docs[i]
    text = line.encode("utf-8")
    url_segment = base_url + "/api/v1.0/segment/?" + "lang=" + quote(lang) + "&text=" + quote(text)
    web = requests.get(url_segment, cookies = conn)
    tuples = json.loads(web.text)
    # print (tuples)
    if 'words' in tuples.keys():
        wordsList = tuples['words'] # segmentation json {'words', [w1, w2,...]} return list
        docSplit.append(wordsList)

outfile_path = os.path.join(folder_path, 'docsSplit.txt')
outFile = codecs.open(outfile_path, "w", encoding='utf-8')

for items in docSplit:
    line = " ".join(items)
    outFile.write(line + "\n")
outFile.close()

# Load testRank module
import textrank
percent = 0.25
summary = textrank.rank(docSplit, percent)

summary_path = os.path.join(folder_path, 'docs_summary.txt')
summary_file = codecs.open(summary_path, "w", encoding='utf-8')

for (index, words, score) in summary:
    # words = [str(w) for w in words]
    line =  str(index) + " ".join(words) + str(score)
    summary_file.write(line + "\n")

