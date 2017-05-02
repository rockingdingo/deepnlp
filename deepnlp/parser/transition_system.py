#!/usr/bin/python
# -*- coding:utf-8 -*-
""" 
Arc-Standard Transition System
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals # compatible with python3 unicode coding

import numpy as np
import sys,os
from collections import namedtuple

pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # .../deepnlp/
sys.path.append(pkg_path)

import reader
from reader import DependencyTree
from reader import Transition

class Configuration():
    '''Configuration for parsing system
    '''
    
    def __init__(self, sentence):
        '''Read one sentence and push them to buffer
        '''
        self.UNKNOWN = 0    # unknown index for any node not found, dictionary <UNKNOWN, 0>
        self.NOT_FOUND = -1 # In DependencyTree, if index not found, return -1
        self.ROOT = 0
        
        self.stack = []     # list: initial state [0], ROOT node
        self.buffer = []    # list: index of token, in sequence [1,2,...]
        self.sentence = sentence
        self.tree = DependencyTree()  # storing arcs
        
        for k in range(1, len(sentence.tokens)):
            self.buffer.append(k)                  # e.g. buffer = [1,2,3,...,N]
            self.tree.add(None, None, None, None)  # initialize tree with N empty transitions
        self.stack.append(0)                       # e.g. stack = [0]  Push 0 id of 'ROOT' to stack
        
    def step(self, arc):
        ''' System perform one step according to the arc in the dependency tree
            s1: Top element on stack
            s2: Second Top element on stack
            Left-Arc: L(s1,s2), head is s2, add left arc, remove s1, different than [Chen and Manning 2014.]
            Right-Arc: R(s2,s1), head is s1, add right arc, remove s2
        '''
        # when stack has [ROOT, node] and buffer is not empty, taken shift action
        # when stack has [ROOT, node] and buffer is [], taken last action and remove s
        if (len(self.stack) <= 2 and len(self.buffer) > 0): 
            self.shift()
            return
        
        s1 = self.get_stack(0)
        s2 = self.get_stack(1)
        
        if ('L' in arc):
            label_id = int(arc[2:len(arc)-1]) # L(50) -> 50
            self.set_arc(s1, s2, label_id)    # set the arc of node s1
            self.remove_stack(0)              # remove s1
        elif ('R' in arc):
            label_id = int(arc[2:len(arc)-1]) # R(50) -> 50
            self.set_arc(s2, s1, label_id)    # set the arc of node s2
            self.remove_stack(1)              # remove s2
        else:
            self.shift()
    
    def shift(self):
        if len(self.buffer) == 0:
            return
        k = self.buffer.pop(0)  # pop and remove the first element k in buffer
        self.stack.append(k)      # push to the item k stack
    
    def get_buffer(self, i):
        return self.buffer[i] if i < len(self.buffer) else self.NOT_FOUND
    
    def remove_buffer(self, i):
        size = len(self.buffer)
        if size < (i + 1):
            return self.NOT_FOUND
        else:
            self.buffer.pop(i) 
    
    def get_stack(self, i):
        ''' Get the top ith item from stack, s(0) is top, s(1) is second top
        '''
        size = len(self.stack)
        return self.stack[(size -i - 1)] if i < size else self.NOT_FOUND
    
    def remove_stack(self, i):
        '''Remove the top ith item from stack,
        '''
        size = len(self.stack)
        if (i + 1) > size:
            return None
        else:
            self.stack.pop(size -i - 1)
    
    def add_arc(self, head, label):
        self.tree.add(None, None, head, label)
    
    def set_arc(self, i, head, label):
        ''' i index start at 1, 0 is ROOT
        '''
        self.tree.set(i, None, None, head, label)
    
    def get_word(self, i):  # index start at 1,  0 is ROOT node
        if (i == 0):
            return self.ROOT
        if (i == self.NOT_FOUND):
            return self.UNKNOWN
        return self.sentence.tokens[i].word
    
    def get_tag(self, i):   # index start at 1,  0 is ROOT node
        if (i == 0):
            return self.ROOT
        if (i == self.NOT_FOUND):
            return self.UNKNOWN
        return self.sentence.tokens[i].tag
    
    def get_label(self, i): # index start at 1,  0 is ROOT node
        if (i == 0):
            return self.ROOT      
        if (i == self.NOT_FOUND):
            return self.UNKNOWN
        return self.tree.get_label(i)
    
    def get_left_child(self, k, index):
        ''' Get the left most node, whose head is the kth token;
            index = 1, first leftmost
            index = 2, second leftmost
        '''
        if (k == self.NOT_FOUND):  # k = -1
            return self.NOT_FOUND
        c = 0      # child cound
        for i in range(1, k):
            if (self.tree.get_head(i) == k):  # the head of ith node is k
                c += 1
                if (c == index):
                    return i  # the first node found with head k,e.g. leftmost
        return self.NOT_FOUND
    
    def get_right_child(self, k, index):
        ''' Get the right most node, whose head is the kth token;
            index = 1, first rightmost
            index = 2, second rightmost
        '''
        c = 0
        if (k == self.NOT_FOUND):  # k = -1
            return self.NOT_FOUND
        for i in range(self.tree.count(),k,-1):
            if (self.tree.get_head(i) == k):
                c += 1
                if (c == index):
                    return i # the first node found with head k,e.g. rightmost
        return self.NOT_FOUND
    
    def has_other_child(self, k, tree):
        ''' Compare tree in config and the benchmark tree
            If there are other child of kth node in benchmark tree, whose arc is not added to config yet, 
            Return True, otherwise return False
        '''
        for i in range(1, tree.count()+1):
            if (tree.get_head(i) == k and self.tree.get_head(i) != k):
                return True
        return False
    
    def is_terminal(self):
        '''Check if the configuration is in terminal state
        '''
        if (len(self.stack) == 1 and len(self.buffer) == 0):
            return True
        else:
            return False

def get_features(config):
    '''Get Features from the current configuration
        return: list[int]  word, pos, labels
        unknown features: -1, which has no leftmost/rightmost child
    '''
    # To Do:
    # Update the hard coded feature template
    # Handle Feature -1 option
    # s1,s2,s3, b1,b2,b3
    feat, feat_w, feat_p, feat_l= [], [], [], [] # feature, feature word, feature pos, feature label
    
    # Stack: s1, s2, s3
    for i in range(3):
        idx = config.get_stack(i)
        word_id = config.get_word(idx)
        pos_id = config.get_tag(idx)
        feat_w.append(word_id)
        feat_p.append(pos_id)
    
    # Buffer: b1, b2, b3
    for i in range(3):
        idx = config.get_buffer(i)
        word_id = config.get_word(idx)
        pos_id = config.get_tag(idx)
        feat_w.append(word_id)
        feat_p.append(pos_id)
    
    # Leftmost and Rightmost children of top k word on stack, lc(si), rc(si)
    # Leftmost of the leftmost/ rightmost of the right most on the stack, lc(lc(si)), rc(rc(si))
    for i in range(2):  # s1, s2
        k = config.get_stack(i)
        # first leftmost child
        lc1 = config.get_left_child(k, 1)
        feat_w.append(config.get_word(lc1))
        feat_p.append(config.get_tag(lc1))
        feat_l.append(config.get_label(lc1))
        
        # second leftmost child
        lc2 = config.get_left_child(k, 2)
        feat_w.append(config.get_word(lc2))
        feat_p.append(config.get_tag(lc2))
        feat_l.append(config.get_label(lc2))
        
        # first rightmost child
        rc1 = config.get_right_child(k, 1)
        feat_w.append(config.get_word(rc1))
        feat_p.append(config.get_tag(rc1))
        feat_l.append(config.get_label(rc1))
    
        # second rightmost child
        rc2 = config.get_right_child(k, 2)
        feat_w.append(config.get_word(rc2))
        feat_p.append(config.get_tag(rc2))
        feat_l.append(config.get_label(rc2))
        
        # leftmost of the leftmost child
        lc_lc1 = config.get_left_child(lc1, 1)
        feat_w.append(config.get_word(lc_lc1))
        feat_p.append(config.get_tag(lc_lc1))
        feat_l.append(config.get_label(lc_lc1))
        
        # rightmost of the rightmost child
        rc_rc1 = config.get_left_child(rc1, 1)
        feat_w.append(config.get_word(rc_rc1))
        feat_p.append(config.get_tag(rc_rc1))
        feat_l.append(config.get_label(rc_rc1))
        
    feat.extend(feat_w) # feat_w: 18 features
    feat.extend(feat_p) # feat_p: 18 features
    feat.extend(feat_l) # feat_l: 12 features
    return feat

def get_next_arc(config, tree):
    ''' Predict the next arc decision based on current configuration and the benchmark in the tress
        Return arc label id
    '''
    s1 = config.get_stack(0)   # s1: top element on the stack
    s2 = config.get_stack(1)   # s2: second top element on the stack
    label = ""
    if (tree.get_head(s1) == s2 and (not config.has_other_child(s1, tree))):  # Left Arc: e.g. L(5)
        label = "L(" + str(tree.get_label(s1)) + ")"
    elif (tree.get_head(s2) == s1):                                           # Right Arc: e.g. R(5)
        label = "R(" + str(tree.get_label(s2)) + ")"
    else:                               
        label = "S"                                                           # Shift
    return label

def generate_arcs(label_dict):
    ''' 
        Args: <label, id>
        Return: 'L(label_id)' 'R(label_id)' 'S', total (2*Nl+1) arc_labels
            e.g. L(0), R(0), L(1), R(1), ..., 'S'
    '''
    arc_labels = {}
    size = len(label_dict)
    idx = 0
    for i in range(size):
        arc_labels[("L(" + str(i) + ")" )] = idx
        idx += 1
        arc_labels[("R(" + str(i) + ")" )] = idx
        idx += 1
    arc_labels['S'] = idx
    return arc_labels

def generate_examples(sents, trees, batch_size, label_dict):
    ''' main function for generating training examples
        Args:
            sents: list[Sentence]
            trees: list[DependencyTree]
            arc_labels: dict, <K,V> <label,id>
        Return:
            iterator over (X, Y)
    '''
    arc_labels = generate_arcs(label_dict)   # （L(1), 1) (L(2), 2)
    label_num = len(arc_labels)
    X, Y = [], []

    print ("Generating examples for %d sentences in total ..." % len(sents))

    for idx, sent in enumerate(sents):
        if idx % 100 == 0:
            print("Generating examples for sentence %d" % idx)
        tree = trees[idx]
        config = Configuration(sent)
        # iterate over transition based system
        while not config.is_terminal():
            features = get_features(config) # 1-D list
            next_arc = get_next_arc(config, tree)  # number
            next_arc_id = arc_labels[next_arc]
            # append X features
            X.append(features)
            # append Y one hot vectors
            y_one_hot = np.zeros(label_num)
            y_one_hot[next_arc_id] = 1.0
            Y.append(y_one_hot)
            # Configuration Take One Step 
            config.step(next_arc)
    
    # print feature number
    # print ("Feature dimension %d" % len(X[0]))

    X = np.array(X) # convert list to np array
    Y = np.array(Y) # convert list to np array
    #print ("X shape %d , %d" % (X.shape[0], X.shape[1]))
    #print ("Y shape %d , %d" % (Y.shape[0], Y.shape[1]))

    batch_len = X.shape[0] // batch_size   # number of training batches generated
    # print ("batch len is %d" % batch_len)

    for i in range(batch_len):
        x = np.array(X[i*batch_size:(i+1)*batch_size, :]) # shape [batch_size, feature_size]
        y = np.array(Y[i*batch_size:(i+1)*batch_size, :]) # shape [batch_size]
        yield (x, y)

def test():
    data_path ="./data/zh"
    print ("Data Path: " + data_path)
    train_sents, train_trees, dev_sents, dev_trees, vocab_dict, pos_dict, label_dict = reader.load_data(data_path)
    
    print ("Vocab Dict Size %d" % len(vocab_dict))
    print ("POS Dict Size %d" % len(pos_dict))
    print ("Label Dict Size %d" % len(label_dict))  # unique labels size, Nl, not arc label num
    
    train_dataset = generate_examples(train_sents, train_trees, 1, label_dict)   # Unknown feature index
    for step, (x, y) in enumerate(train_dataset):
        if (step <= 10):
            print ("Step id: %d" % step)
            print (x)
            print (y)
        else:
            break

if __name__ == "__main__":
    test()
