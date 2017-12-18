#!/usr/bin/python
# -*- coding:utf-8 -*-
# B, M, E, S: Beginning, Middle, End, Single 4 tags

import sys,os
import CRFPP
from model_util import registered_models

# linear chain CRF model path, need str input, convert unicode to str in python2, <str> object in python3
pkg_path = os.path.dirname(os.path.abspath(__file__))
version_info = "py3" if (sys.version_info>(3,0)) else "py2"
DEFAULT_MODEL = str(os.path.join(pkg_path, "segment/models/zh/crf_model"))

class Tokenizer(object):

    def __init__(self, model_path = DEFAULT_MODEL):
        self.model = CRFPP.Tagger("-m " + model_path)
    
    def seg(self, text):
        '''
        text: String, text to be segmented;
        model: path of pretrained CRFPP model,
        '''
        segList = []
        model = self.model
        model.clear()
        for char in text.strip(): # char in String
            char = char.strip()
            if char:
                input_char = (char + "\to\tB").encode('utf-8') if (version_info == "py2") else (char + "\to\tB")
                model.add(input_char)
        model.parse()
        size = model.size()
        xsize = model.xsize()
        word = ""
        for i in range(0, size):
            for j in range(0, xsize):
                char = model.x(i, j).decode('utf-8') if (version_info == "py2") else model.x(i, j)
                tag = model.y2(i)
                if tag == 'B':
                    word = char
                elif tag == 'M':
                    word += char
                elif tag == 'E':
                    word += char
                    segList.append(word)
                    word = ""
                else: # tag == 'S'
                    word = char
                    segList.append(word)
                    word = ""
        return segList

def load_model(name = 'zh'):
    ''' model_path e.g.: ./segment/models/zh/crf_model
        Loadg pretrained subfield models...
    '''
    registered_model_list = registered_models[0]['segment']
    if name not in registered_model_list:
        print ("WARNING: Input model name '%s' is not registered..." % name)
        print ("WARNING: Please register the name in model_util.registered_models...")
        return None
    model_path = str(os.path.join(pkg_path, "segment/models/", name, "crf_model"))
    if os.path.exists(model_path):
        print ("NOTICE: Loading model from below path %s..." % model_path)
        return Tokenizer(model_path)
    else:
        print ("WARNING: Input model path %s doesn't exist ..." % model_path)
        print ("WARNING: Please download model file using method: deepnlp.download(module='%s', name='%s')" % ('segment', name))
        print ("WARNING: Loading default model %s..." % DEFAULT_MODEL)
        return Tokenizer(DEFAULT_MODEL)

def load_user_model(model_path):
    ''' model_path e.g.: ./segment/models/zh/crf_model
    '''
    if os.path.exists(model_path):
        print ("NOTICE: Loading model from below path %s..." % model_path)
        return Tokenizer(model_path)
    else:
        print ("WARNING: Input model path %s doesn't exist, please download model file using deepnlp.download() method" % name)        
        print ("WARNING: Loading default model %s..." % DEFAULT_MODEL)
        return Tokenizer(DEFAULT_MODEL)
