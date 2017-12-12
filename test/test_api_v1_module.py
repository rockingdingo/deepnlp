#coding:utf-8
'''
Demo for calling API of deepnlp.org web service
Anonymous user of this package have limited access on the number of API calling 100/day
Please Register and Login Your Account to deepnlp.org to get unlimited access to fully support
api_service API module, now supports both windows and linux platforms.
'''

from __future__ import unicode_literals

import json, requests, sys, os
if (sys.version_info>(3,0)): from urllib.parse import quote 
else : from urllib import quote

from deepnlp import api_service
login = api_service.init()          # registration, if failed, load default empty login {} with limited access
login = {}                          # use your personal login {'username': 'your_user_name' , 'password': 'your_password'}
conn = api_service.connect(login)   # save the connection with login cookies

# API Setting
text = ("我爱吃北京烤鸭").encode('utf-8')  # convert text from unicode to utf-8 bytes, quote() function

# Segmentation
url_segment = "http://www.deepnlp.org/api/v1.0/segment/?" + "lang=zh" + "&text=" + quote(text)
web = requests.get(url_segment, cookies = conn)
tuples = json.loads(web.text)
wordsList = tuples['words'] # segmentation json {'words', [w1, w2,...]} return list
print ("Segmentation API:")
print (" ".join(wordsList).encode("utf-8"))

# POS tagging
url_pos = "http://www.deepnlp.org/api/v1.0/pos/?"+ "lang=zh" + "&text=" + quote(text)
web = requests.get(url_pos, cookies = conn)
tuples = json.loads(web.text)
pos_str = tuples['pos_str'] # POS json {'pos_str', 'w1/t1 w2/t2'} return string
print ("POS API:")
print (pos_str.encode("utf-8"))

# NER tagging
url_ner = "http://www.deepnlp.org/api/v1.0/ner/?" + "lang=zh" + "&text=" + quote(text)
web = requests.get(url_ner, cookies = conn)
tuples = json.loads(web.text)
ner_str = tuples['ner_str'] # NER json {'ner_str', 'w1/t1 w2/t2'} return list
print ("NER API:")
print (ner_str.encode("utf-8"))

# Pipeline
annotators = "segment,pos,ner"
url_pipeline = "http://www.deepnlp.org/api/v1.0/pipeline/?" + "lang=zh" + "&text=" + quote(text) + "&annotators=" + quote(annotators)
web = requests.get(url_pipeline, cookies = conn)
tuples = json.loads(web.text)
segment_str = tuples['segment_str']  # segment module
pos_str = tuples['pos_str']   # pos module
ner_str = tuples['ner_str']   # ner module
ner_json = tuples['ner_json'] # ner result in json

# output
def json_to_str(json_dict):
    json_str = ""
    for k, v in json_dict.items():
        json_str += ("'" + k + "'" + ":" + "'" + v + "'" + ",")
    json_str = "{" + json_str + "}"
    return json_str

print ("Pipeline API:")
print (segment_str.encode("utf-8"))
print (pos_str.encode("utf-8"))
print (ner_str.encode("utf-8"))
print ("NER JSON:")
print (json_to_str(ner_json).encode("utf-8"))

