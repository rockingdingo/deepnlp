#!/usr/bin/python
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals # compatible with python3 unicode

import codecs
import pickle

def gen_trie_dict(file_path, encoding = 'utf-8'):
    """ Args: input format word \t prop
    """
    trie = {}
    f = codecs.open(file_path, encoding=encoding)
    for line in f:
        items = line.replace("\n","").split("\t")
        word = None
        prop = None
        if (len(items) == 1):   # word
            word = items[0]
        elif (len(items) == 2): # word \t prop
            word = items[0]
            prop = items[1]
        else:
            continue
        tmp = trie
        if (word):
            for char in word:
                if char not in tmp:
                    tmp[char] = {}
                tmp = tmp[char]   # move to child
            # Set Property
            if (prop):
                try:
                    if 'prop' in tmp:
                        tmp['prop'].append(prop)  # set property to the last char of the word
                    else:
                        tmp['prop'] = [prop]
                except Exception as e:
                    print (e.message)
            else:
                try:
                    tmp['prop'] = None
                except Exception as e:
                    print (e.message)
    return trie

def gen_prefix_dict(file_path, encoding = 'utf-8'):
    """ Args: Generate Prefix Dictionary
    """
    prefix_dict = {}
    f = codecs.open(file_path, encoding=encoding)
    for line in f:
        items = line.replace("\n","").split("\t")
        word = None
        propList = []   # empty list
        prop = None
        if (len(items) == 1):
            word = items[0]
        elif (len(items) == 2):
            word = items[0]
            prop = items[1]
        else:
            continue
        if (word):
            # Add all prefix to current word
            for idx in xrange(len(word)):       
                curSeg = word[0:(idx+1)]
                if curSeg not in prefix_dict:  
                    prefix_dict[curSeg] = None
            # Add prop to list of the current word dict  <K,V> <word, List<string> props>
            if word in prefix_dict:
                if prefix_dict[word] is not None:  # prefix props is None
                    prefix_dict[word].append(prop)
                else:
                    prefix_dict[word] = [prop]
            else:
                prefix_dict[word] = [prop]
    f.close()
    return prefix_dict

def test():
    dictPath = "./ner/dict/zh_o2o/entity_tags.dic"
    word = "星巴克"
    # Test Trie Dict
    trie = gen_trie_dict(dictPath)
    # test search trie
    tmp = trie
    tag = None
    for ch in word:
        if ch in tmp.keys():
            tmp = tmp[ch]
        else:
            print ("not in trie dict")
            break
    if "prop" in tmp.keys():
        tags = tmp["prop"]
        print (tags)

    # Test pdDict
    pd = gen_prefix_dict(dictPath)
    if word in pd.keys():
        print (pd[word])

def gen_pickle_file():
    dict_path = "./ner/dict/zh_o2o/entity_tags.dic"
    print ("NOTICE: Start Loading Default Entity Tag Dictionary...")
    prefix_dict = gen_prefix_dict(dict_path)

    print ("NOTICE: prefix_dict Size %d" % len(prefix_dict))
    
    pickle_file = "./ner/dict/zh_o2o/entity_tags.dic.pkl"
    fw = open(pickle_file, 'wb')
    pickle.dump(prefix_dict, fw, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    #test()
    gen_pickle_file()
