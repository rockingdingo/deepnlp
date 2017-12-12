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
import pickle
import re
from collections import namedtuple

pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # .../deepnlp/.
sys.path.append(pkg_path)

from parse import reader
from parse.reader import DependencyTree
from parse.reader import Transition
from parse.reader import save_instance
from parse.reader import load_instance

#Constant
MAX_CONFIG_STEP = 50  # maximum number of step moved by the transition configuration, avoid illegal tree 

class Configuration():
    '''Configuration for parsing system
    '''
    
    def __init__(self, sentence):
        '''Read one sentence and push them to buffer
        '''
        self.NOT_FOUND = -1 # If index not found, return -1
        self.UNKNOWN = 0    # Word or Tag index for any unknown node, dictionary <UNKNOWN, 0>
        self.ROOT = 0       # Word and Tag Index of ROOT Node
        
        self.stack = []     # list: initial state [0], ROOT node
        self.buffer = []    # list: index of token, in sequence [1,2,...]
        self.sentence = sentence
        self.tree = DependencyTree()  # storing arcs
        
        for k in range(1, len(sentence.tokens)):
            self.buffer.append(k)                  # tokens list size: (N + 1), buffer = [1,2,3,...,N]
            self.tree.add(None, None, None, None)  # initialize tree with N empty transitions
        self.stack.append(self.ROOT)               # e.g. stack = [0]  Push 0 id of 'ROOT' to stack
        
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
        # Get Top two element on stack top
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


### REGEX for Extracting Feature Template
RE_STACK = r"STACK\[([^\]]+)\]"                                             # STACK[i]
RE_BUFFER = r"BUFFER\[([^\]]+)\]"                                           # BUFFER[i]
RE_LC_STACK = r"LC\[([^\]]+)\]\(STACK\[([^\]]+)\]\)"                        # LC[i](STACK[i])
RE_RC_STACK = r"RC\[([^\]]+)\]\(STACK\[([^\]]+)\]\)"                        # RC[i](STACK[i])
RE_LC_LC_STACK = r"LC\[([^\]]+)\]\(LC\[([^\]]+)\]\(STACK\[([^\]]+)\]\)\)"   # LC[i](LC[i](STACK[i]))
RE_RC_RC_STACK = r"RC\[([^\]]+)\]\(RC\[([^\]]+)\]\(STACK\[([^\]]+)\]\)\)"   # RC[i](RC[i](STACK[i]))

def get_features(config, feature_tpls):
    """ Args: config: current sentence Configuration
        feature_tpls: list of storing feature templates   type_position e.g. WORD_STACK[0]
    """
    feat, feat_w, feat_p, feat_l= [], [], [], [] # feature, feature word, feature pos, feature label

    for feat in feature_tpls:
        items = feat.split("_")
        if (len(items) == 2):
            feat_type_str = items[0]       # WORD, POS, DEPREL
            feat_position_str = items[1]
            # 匹配feat_position
            feat_position = int(get_feature_position(config, feat_position_str))
            if (feat_type_str == 'WORD'):
                word_id = config.get_word(feat_position)
                feat_w.append(word_id)
            elif (feat_type_str == 'POS'):
                pos_id = config.get_tag(feat_position)
                feat_p.append(pos_id)
            elif (feat_type_str == 'DEPREL'):
                deprel_id = config.get_label(feat_position)
                feat_l.append(deprel_id)
            else:
                print ("DEBUG: Current Feature Type String is not found...%s" % feat_type_str)
    # Combine Features
    feat = feat_w
    feat.extend(feat_p) # feat_p: 18 features
    feat.extend(feat_l) # feat_l: 12 features
    #print ("DEBUG: Feature Total Number is %d" % len(feat))
    return feat

def get_feature_position(config, feat):
    """ Args: 
            config: Configuration of current config
            tpl: input feature template
        Return: word id of tpl position
    """
    word_id = 0
    if(re.match(RE_STACK, feat)):      # STACK[i]
        groups = re.findall(RE_STACK, feat)
        top = int(groups[0]) if (len(groups) > 0) else 0
        word_id = config.get_stack(top)
        return word_id
    elif (re.match(RE_BUFFER, feat)):   # BUFFER[i]
        groups = re.findall(RE_BUFFER, feat)
        top = int(groups[0]) if (len(groups) > 0) else 0
        word_id = config.get_buffer(top)
        return word_id
    elif (re.match(RE_LC_STACK, feat)):   # LC[i]_STACK[i]
        groups = re.findall(RE_LC_STACK, feat)[0]
        if (len(groups) == 2):
            lc_id = int(groups[0])   # start at 0
            stack_id = int(groups[1])
            word_id = config.get_left_child(config.get_stack(stack_id), lc_id + 1)
            return word_id
    elif (re.match(RE_RC_STACK, feat)):   # RC[i]_STACK[i]
        groups = re.findall(RE_RC_STACK, feat)[0]
        if (len(groups) == 2):
            rc_id = int(groups[0])   # start at 0
            stack_id = int(groups[1])
            word_id = config.get_right_child(config.get_stack(stack_id), rc_id + 1)
            return word_id
    elif (re.match(RE_LC_LC_STACK, feat)):   # LC[i](LC[i](STACK[i]))
        groups = re.findall(RE_LC_LC_STACK, feat)[0]
        if (len(groups) == 3):
            lc_id_1 = int(groups[0])   # start at 0
            lc_id_2 = int(groups[1])   # start at 0
            stack_id = int(groups[2])
            word_id = config.get_left_child(config.get_left_child(config.get_stack(stack_id), lc_id_2 + 1), lc_id_1 + 1)
            return word_id
    elif (re.match(RE_RC_RC_STACK, feat)):   # RC[i](RC[i](STACK[i]))
        groups = re.findall(RE_RC_RC_STACK, feat)[0]
        if (len(groups) == 3):
            rc_id_1 = int(groups[0])   # start at 0
            rc_id_2 = int(groups[1])   # start at 0
            stack_id = int(groups[2])
            word_id = config.get_right_child(config.get_right_child(config.get_stack(stack_id), rc_id_2 + 1), rc_id_1 + 1)
            return word_id
    else:
        print ("DEBUG: Current Input Template Doesn't Match Any position...")
        return word_id

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

def generate_examples(sents, trees, label_dict, feature_tpl, instance_path, is_train = True):
    ''' main function for generating training examples
        Args:
            sents: list[Sentence]
            trees: list[DependencyTree]
            arc_labels: dict, <K,V> <label,id>
            feature_tpl: list: Feature Template
            instance_path: path of saved instance
        Return:
            iterator over (X, Y)
    '''
    # Check if existing examples are already saved
    X_instance_path = ""
    Y_instance_path = ""
    if (is_train):
        X_instance_path = os.path.join(instance_path, "train_examples_X.pkl")
        Y_instance_path = os.path.join(instance_path, "train_examples_Y.pkl")
    else:
        X_instance_path = os.path.join(instance_path, "dev_examples_X.pkl")
        Y_instance_path = os.path.join(instance_path, "dev_examples_Y.pkl")

    if os.path.exists(X_instance_path) and os.path.exists(Y_instance_path):
        print ("NOTICE: Restoring Examples X from %s" % X_instance_path)
        print ("NOTICE: Restoring Examples Y from %s" % Y_instance_path)
        X = load_instance(X_instance_path)
        Y = load_instance(Y_instance_path)
        return X, Y
    else:
        arc_labels = generate_arcs(label_dict)   # （L(1), 1) (L(2), 2)
        label_num = len(arc_labels)
        X, Y = [], []
        print ("Generating examples for %d sentences in total ..." % len(sents))

        for idx, sent in enumerate(sents):
            if idx % 1000 == 0:
                print("Generating examples for sentence %d" % idx)
            tree = trees[idx]
            config = Configuration(sent)
            # iterate over transition based system
            trans_count = 0
            while (not config.is_terminal() and trans_count <= MAX_CONFIG_STEP):
                features = get_features(config, feature_tpl)   # 1-D list
                next_arc = get_next_arc(config, tree)          # number
                next_arc_id = arc_labels[next_arc]
                # append X features
                X.append(features)
                # append Y one hot vectors
                y_one_hot = np.zeros(label_num)
                y_one_hot[next_arc_id] = 1.0
                Y.append(y_one_hot)
                # Configuration Take One Step 
                config.step(next_arc)
                trans_count += 1   # Check Validity avoid infinity loop
        print ("NOTICE: Total Generated Example Count %d" % len(X))
        X = np.array(X) # convert list to np array
        Y = np.array(Y) # convert list to np array
        print ("DEBUG: Saved Example Instance Path %s" % instance_path)
        save_instance(X_instance_path, X)
        save_instance(Y_instance_path, Y)
        return X, Y

def iter_examples(X, Y, batch_size):
    """ Args: X: numpy array of [num_example, num_feature_dim]
              Y: numpy array of [num_example, num_target_output]
    """
    batch_len = X.shape[0] // batch_size   # number of training batches generated
    print ("NOTICE: Total Generated Batch Length Count %d" % batch_len)
    for i in range(batch_len):
        x = np.array(X[i*batch_size:(i+1)*batch_size, :]) # shape [batch_size, feature_size]
        y = np.array(Y[i*batch_size:(i+1)*batch_size, :]) # shape [batch_size]
        yield (x, y)

def test():
    data_path ="./data"
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
