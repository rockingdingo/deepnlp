#coding:utf-8
'''
Demo for calling API of deepnlp.org web service
Anonymous user of this package have limited access on the number of API calling 100/day
Please Register and Login Your Account to deepnlp.org to get unlimited access to fully support
api_service API module, now supports both windows and linux platforms.
'''

from __future__ import unicode_literals

import json, requests, sys, os, codecs
if (sys.version_info>(3,0)): from urllib.parse import quote 
else : from urllib import quote

from deepnlp import api_service
login = api_service.init()          # registration, if failed, load default empty login {} with limited access
login = {}                          # use your personal login {'username': 'your_user_name' , 'password': 'your_password'}
conn = api_service.connect(login)   # save the connection with login cookies

# API setting
base_url = 'http://www.deepnlp.org'
lang = 'zh'
annotators = "segment,pos"

# Read text file docs_api.txt
path = os.path.dirname(os.path.abspath(__file__)) # current folder path
filename = os.path.join(path, "docs_api.txt")
file = codecs.open(filename, encoding='utf-8')
docs = []
for line in file:
    if (len(line.strip())!=0):
        docs.append(line)

# Calling Pipeline API
for line in docs:
    text = line.encode("utf-8") # convert text from unicode to utf-8 bytes
    url_pipeline = "http://www.deepnlp.org/api/v1.0/pipeline/?" + "lang=zh" + "&annotators=segment,pos" + "&text="  + quote(text)
    web = requests.get(url_pipeline, cookies = conn)
    tuples = json.loads(web.text) # convert string to json format
    pos_str = tuples['pos_str']   # extract {'pos_str': 'results'} from json tuples
    print (pos_str.encode('utf-8'))

