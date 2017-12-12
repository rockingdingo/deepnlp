#!/usr/bin/python
# -*- coding:utf-8 -*-

from __future__ import unicode_literals # compatible with python3 unicode coding

from deepnlp import nn_parser
parser = nn_parser.load_model(name = 'zh')

#Example 1, Input Words and Tags Both
words = ['它', '熟悉', '一个', '民族', '的', '历史']
tags = ['r', 'v', 'm', 'n', 'u', 'n']

#Parsing
dep_tree = parser.predict(words, tags)
print ("DEBUG: Dependency Tree is:")
print ("id" + "\t" + "form" + "\t" + "lemma" + "\t" + "pos" + "\t" + "ppos"+ "\t" + "head" + "\t" + "deprel")
print (dep_tree)

#Fetch result from Transition Namedtuple
num_token = dep_tree.count()
print ("id\tword\tpos\thead\tlabel")
for i in range(num_token):
    cur_id = int(dep_tree.tree[i+1].id)
    cur_form = str(dep_tree.tree[i+1].form)
    cur_pos = str(dep_tree.tree[i+1].pos)
    cur_head = str(dep_tree.tree[i+1].head)
    cur_label = str(dep_tree.tree[i+1].deprel)
    print ("%d\t%s\t%s\t%s\t%s" % (cur_id, cur_form, cur_pos, cur_head, cur_label))

#Results
#word id: 1, word : 它, head id: 2, label: SBV
#word id: 2, word : 熟悉, head id: 0, label: HED
#word id: 3, word : 一个, head id: 4, label: QUN
#word id: 4, word : 民族, head id: 5, label: DE
#word id: 5, word : 的, head id: 6, label: ATT
#word id: 6, word : 历史, head id: 2, label: VOB

#Example 2
words = ['这种', '局面', '的', '形成', '有着', '复杂', '的', '社会', '背景']
dep_tree = parser.predict(words)
print ("DEBUG: Dependency Tree is:")
print ("id" + "\t" + "form" + "\t" + "lemma" + "\t" + "pos" + "\t" + "ppos"+ "\t" + "head" + "\t" + "deprel")
print (dep_tree)

# Result
num_token = dep_tree.count()
print ("id\tword\tpos\thead\tlabel")
for i in range(num_token):
    cur_form = str(dep_tree.tree[i+1].form)
    cur_pos = str(dep_tree.tree[i+1].pos)      
    cur_head = str(dep_tree.tree[i+1].head)
    cur_label = str(dep_tree.tree[i+1].deprel)
    print ("%d\t%s\t%s\t%s\t%s" % (i+1, cur_form, cur_pos, cur_head, cur_label))

#Results
#word id: 1, word : 这种, head id: 2, label: ATT
#word id: 2, word : 局面, head id: 3, label: DE
#word id: 3, word : 的, head id: 4, label: ATT
#word id: 4, word : 形成, head id: 5, label: SBV
#word id: 5, word : 有着, head id: 0, label: HED
#word id: 6, word : 复杂, head id: 7, label: DE
#word id: 7, word : 的, head id: 9, label: ATT
#word id: 8, word : 社会, head id: 9, label: ATT
#word id: 9, word : 背景, head id: 5, label: VOB
