#!/usr/bin/python
# -*- coding:utf-8 -*-

"""Utilities for reading parser train.conll, dev.conll files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals # compatible with python3 unicode

from collections import namedtuple
from collections import Counter
import sys, os
import codecs
import re

import numpy as np

global UNKNOWN, DELIMITER
UNKNOWN = "*"
DELIMITER = "\s+" # line delimiter

class TaggedWord(object):
    _word = 0
    _tag = 0
    
    def __init__(self, word, tag):
        self._word, self._tag = word, tag
    
    @property
    def word(self):
        return self._word
    
    @word.setter
    def word(self, word):
        self._word = word
    
    @property
    def tag(self):
        return self._tag
    
    @tag.setter
    def tag(self, tag):
        self._tag = tag

class Sentence(object):
    ''' List[TaggedWord]  
        First element of sentence is 'ROOT' token, actual word/tag pair index start from 1
    '''
    
    def __init__(self):
        self._tokens = []
        self._tokens.append(TaggedWord(0, 0))   # ROOT Node, word index and tag index set to (0,0)
        
    def add(self, word, tag):
        self._tokens.append(TaggedWord(word, tag))
    

    def get_word(self, i):     # starting from 0 ROOT node
        return self._tokens[i].word

    def get_tag(self, i):
        return self._tokens[i].tag

    @property
    def tokens(self):
        return self._tokens
    
    def __repr__(self):
        out = ""
        for t in self._tokens:
            out += (str(t.word) + "/" + str(t.tag) + " ")
        return out

# See CONLL 2006/2009 data format for details
# ID FORM LEMMA POS PPOS _ HEAD DEPREL _ _
# Not all columns are needed for specific language, see details for your choozen one
Transition = namedtuple("Transition", ["id", "form", "lemma", "pos", "ppos", "head", "deprel"])

class DependencyTree():
    ''' List[Transition]
        First element of sentence is 'ROOT' transition, actual transition index start from 1
        Input sentence: w1,w2,...wn
        ID FORM LEMMA POS PPOS _ HEAD DEPREL _ _
        0  ROOT  ROOT None  None  _ 0 _ _
        1   w1  w1 pos1 pos1 _ 4 l1 _ _
        ...
    '''
    row_id = 0
    def __init__(self):
        self.tree = [] # list of transitions
        self.tree.append(Transition(0, 'ROOT', 'ROOT', None, None, None, None))
    
    def add(self, word, pos, head, deprel):
        ''' Add import features to tree
        '''
        self.row_id += 1
        self.tree.append(Transition(self.row_id, word, word, pos, pos, head, deprel))
        
    def set(self, k, word, pos, head, deprel):
        '''Set word , pos, head and label to the kth node, k beginns at 1, 0 is ROOT node
        '''
        if (k >= len(self.tree)):
            raise Exception("k is out of index of head and label list")
            return
        self.tree[k] = Transition(k, word, word, pos, pos, head, deprel)
    
    def get_head(self, k):
        '''Return: int, the head word index of the kth word
        '''
        if (self.tree[k].head):   # Not None
            return int(self.tree[k].head)
        else:
            return None
        
    def get_label(self, k):
        '''Return: int, the label index of the kth word 
        '''
        if (self.tree[k].deprel):   # Not None
            return int(self.tree[k].deprel)
        else:
            return None
    
    def get_root(self):
        ''' Get the index of the node, which has head as 'ROOT'
        '''
        for k in range(len(self.tree)):
            if(self.tree[k].head == 0): # kth node has head as id 0: 'ROOT'
                return k
        return 0
    
    def count(self):
        '''Return the number of Transition in the tree, index start from 1, excluding ROOT node
        '''
        return (len(self.tree) - 1)
    
    def __repr__(self):
        out = ""
        for t in self.tree:
            out += (str(t.id) + "\t" 
                    + str(t.form) + "\t" 
                    + str(t.lemma) + "\t" 
                    + str(t.pos) + "\t" 
                    + str(t.ppos) + "\t"
                    + str(t.head) + "\t"
                    + str(t.deprel) + "\n")
        return out

def _read_file(filename):
    ''' Read the transitions into the program, empty line means end of each sentence
    '''
    transitions = []  # list[list[transitions]]
    file = codecs.open(filename, encoding='utf-8')
    trans = []        # empty list of tuples
    for line in file:
        if (len(line.strip())==0):
            transitions.append(trans)
            trans = []
        else:
            items = line.split("\t")
            if (len(items) == 10):      # id, form, lemma, pos, ppos, _ , head, deprel, _ , _
                trans.append(Transition(items[0],items[1],items[2],items[3], items[4],items[6],items[7]))
    
    if (len(trans)>0):   # append the last sentence
        transitions.append(trans)
    
    count = len(transitions)
    print ("Number of sentences read: %d" % count)
    return transitions

def _gen_index_dict(l):
    ''' Generate index dictionary based on the <key> and <the sorted order index>
        UNKNOWN: <UNKNOWN, 0> always has index 0
        Args: dict <K,V> key, count
        Returns: dict <K,V> key, id
    '''
    counter = Counter(l)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    sorted_list, _ = list(zip(*count_pairs))
    idx_dict = dict(zip(sorted_list, range(1, len(sorted_list) + 1)))  # idx 1:n
    idx_dict[UNKNOWN] = 0
    return idx_dict

def _build_dict(transitions):
    vocabs = [] # word
    pos_tags = []   # pos tags
    labels = [] # labels
    
    for trans in transitions:
        for t in trans:
            vocabs.append(t.form)      # word
            pos_tags.append(t.pos)     # pos
            labels.append(t.deprel)    # dep relations labels
    
    # Sort the keys and generate index
    vocab_dict = _gen_index_dict(vocabs)
    pos_dict = _gen_index_dict(pos_tags)
    label_dict = _gen_index_dict(labels)
    return vocab_dict, pos_dict, label_dict

def _get_tree(transitions, vocab_dict, pos_dict, label_dict):
    ''' Generate dependency trees, items are the index of vocab, pos and label
        Return: list[DependencyTree]
    '''
    trees = [] # list of dependency trees
    for trans in transitions:
        tree = DependencyTree()
        for t in trans:
            # word_index, pos_index, head_id, label_index
            word_idx = vocab_dict[t.form] if t.form in vocab_dict.keys() else vocab_dict[UNKNOWN]
            pos_idx = pos_dict[t.pos] if t.pos in pos_dict.keys() else pos_dict[UNKNOWN]
            label_idx = label_dict[t.deprel] if t.deprel in label_dict.keys() else label_dict[UNKNOWN]
            tree.add(word_idx, pos_idx, t.head, label_idx)
        trees.append(tree)
    return trees

def _get_sentence(transitions, vocab_dict, pos_dict):
    '''Return list[Sentence]   word/tag pairs, first node is ROOT index 0, actual tokens start at index 1
    '''
    sentences = []
    for trans in transitions:
        sentence = Sentence()
        root_word = vocab_dict[UNKNOWN]
        root_pos = pos_dict[UNKNOWN]
        sentence.add(root_word, root_pos) # adding Root Node to sentences
        for t in trans:
            word_idx = vocab_dict[t.form] if t.form in vocab_dict.keys() else vocab_dict[UNKNOWN]
            pos_idx = pos_dict[t.pos] if t.pos in pos_dict.keys() else pos_dict[UNKNOWN]
            sentence.add(word_idx, pos_idx)
        sentences.append(sentence)
    return sentences

def _tokenize_data(transitions, vocab_dict, pos_dict, label_dict):
    trees = [] # list of dependency trees
    sentences = []
    count = 0
    for trans in transitions:
        if (count % 1000 == 0):
            print ("Tokenizing Data Line %d ...." % count)
        tree = DependencyTree()
        sentence = Sentence()
        for t in trans:
            # word_index, pos_index, head_id, label_index
            word_idx = vocab_dict[t.form] if t.form in vocab_dict.keys() else vocab_dict[UNKNOWN]
            pos_idx = pos_dict[t.pos] if t.pos in pos_dict.keys() else pos_dict[UNKNOWN]
            label_idx = label_dict[t.deprel] if t.deprel in label_dict.keys() else label_dict[UNKNOWN]
            tree.add(word_idx, pos_idx, t.head, label_idx)
            sentence.add(word_idx, pos_idx)
        trees.append(tree)
        sentences.append(sentence)
        count += 1
    return sentences, trees

def _save_vocab(dict, path):
    # save utf-8 code dictionary
    file = codecs.open(path, "w", encoding='utf-8')
    for k, v in dict.items():
        # k is unicode, v is int
        line = k + "\t" + str(v) + "\n" # unicode
        file.write(line)

def _read_vocab(path):
  # read utf-8 code
    file = codecs.open(path, encoding='utf-8')
    vocab_dict = {}
    for line in file:
        pair = line.replace("\n","").split("\t")
        vocab_dict[pair[0]] = int(pair[1])
    return vocab_dict

def _reverse_map(dic):
    rev_map = {}
    for k, v in dic.items():
        rev_map[v] = k
    return rev_map

def load_data(data_path=None):
    """Load raw training and development data from data directory "data_path".
    Args: 
        data_path
    Returns:
        train_sents, train_trees, dev_sents, dev_trees, vocab_dict, pos_dict, label_dict
    """
  
    train_path = os.path.join(data_path, "train.conll")
    dev_path = os.path.join(data_path, "dev.conll")
    train_transitions = _read_file(train_path)
    dev_transitions = _read_file(dev_path)
    
    # Generate Dictionary from training dataset
    vocab_dict, pos_dict, label_dict = _build_dict(train_transitions)
    print ("Building Vocabulary...")
    print ("Vocab dict size %d ..." % len(vocab_dict))
    print ("POS dict size %d ..." % len(pos_dict))
    print ("Dependency label dict size %d ..." % len(label_dict))

    print ("Saving vocab_dict, pos_dict, label_dict, ...")
    _save_vocab(vocab_dict, os.path.join(data_path, "vocab_dict"))
    _save_vocab(pos_dict, os.path.join(data_path, "pos_dict"))
    _save_vocab(label_dict, os.path.join(data_path, "label_dict"))
    
    # Generate Training Dataset
    print ("Tokenizing Train Data...")
    train_sents, train_trees = _tokenize_data(train_transitions, vocab_dict, pos_dict, label_dict)
    
    # Generate Dev Dataset
    print ("Tokenizing Dev Data...")
    dev_sents, dev_trees = _tokenize_data(dev_transitions, vocab_dict, pos_dict, label_dict)
    
    return (train_sents, train_trees, dev_sents, dev_trees, vocab_dict, pos_dict, label_dict)

def test():
    ''' Sample method for testing
    '''
    data_path ="./data/zh"
    print ("Data Path: " + data_path)
    train_sents, train_trees, dev_sents, dev_trees, vocab_dict, pos_dict, label_dict = load_data(data_path)
    print ("Examples first sentence")
    print (train_sents[0])
    print (train_trees[0])
    
    print (len(train_sents))
    print (len(train_trees))
    print (train_trees[0])
    
if __name__ == '__main__':
    test()
