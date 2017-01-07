#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Evaluation Method for summarization tasks, including BLUE and ROUGE score
"""
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

#from matplotlib import pyplot as plt # drawing heat map of attention weights

def evaluate(X, Y, method = "rouge_n", n = 2):
  score = 0.0
  if (method == "rouge_n") :
    score = eval_rouge_n(X, Y, n)
  elif (method == "rouge_l"):
    score = eval_rouge_l(X, Y)
  elif (method == "bleu"):
    score = eval_bleu(X, Y)
  else:
    score = 0.0
  return score

def eval_bleu(X, Y, n = 2):
  # To Do
  
  
  
  
  
  
  return 0

def eval_rouge_n(y_candidate, y_reference, n = 2):
  '''
  Args: 
    y_candidate: list, fitted line (predicted)
    y_reference: list of list, [[], [],], referenced line (target)
  Return:
    rouge_n score:double, maximum of pairwise rouge-n score
  '''
  if (type(y_reference[0]) != list):
    print ('y_reference should be list of list')
    return
  
  m = len(y_reference)
  rouge_score = []
  ngram_cand = generate_ngrams(y_candidate, n)
  for i in range(m):
    ngram_ref = generate_ngrams(y_reference[i], n)
    num_match = count_match(ngram_cand, ngram_ref)
    rouge_score.append(num_match/len(ngram_ref))
  return max(rouge_score)

def generate_ngrams(input_list, n):
  '''
  zip(x, x[1:,],x[2,],...x[n,]), end with shorted list
  '''
  return zip(*[input_list[i:] for i in range(n)])

def count_match(listA, listB):
  match_list = [tuple for tuple in listA if tuple in listB]
  return len(match_list)

def eval_rouge_l(y_candidate, y_reference):
  '''
  Args: 
    y_candidate: list, fitted line (predicted)
    y_reference: list of list, [[], [],], referenced line (target)
  Return:
    rouge_l score:double, F1 score of longest common sequence
  '''
  if (type(y_reference[0]) != list):
    print ('y_reference should be list of list')
    return
  K = len(y_reference)
  lcs_count = 0.0
  total_cand = len(y_candidate) # total of candidate words
  total_ref = 0.0  # total of reference words
  
  for k in range(K):
    cur_lcs = LCS(y_candidate, y_reference[k])
    lcs_count += len(cur_lcs)
    total_ref += len(y_reference[k])
  
  recall = lcs_count/total_ref
  precision = lcs_count/total_cand
  beta = 8.0 # coefficient
  f1 = (1 + beta * beta) * precision * recall/(recall + beta * beta * precision)
  return f1

def LCS(X, Y):
  '''Get the element of longest common sequence
  '''
  length, flag = calc_LCS(X, Y)
  common_seq_rev = [] # reverse sequence
  # starting from end of X and Y
  start_token = "START"
  X_new = [start_token] + list(X)
  Y_new = [start_token] + list(Y)
  i = len(X_new) - 1
  j = len(Y_new) - 1
  while(i >= 0 and j >= 0):
    if (flag[i][j] == 1):
      common_seq_rev.append(X_new[i])
      i -= 1
      j -= 1
    elif (flag[i][j] == 2):
      i -= 1   # i -> i-1
    else:
      j -= 1   # flag[i][j] == 3, j -> j-1
  common_seq =[common_seq_rev[len(common_seq_rev) - 1 - i] for i in range(len(common_seq_rev))]
  return common_seq

def calc_LCS(X, Y):
  '''
    Calculate Longest Common Sequence
    Get the length[][] matrix and flag[][] matrix of X and Y;
    length[i][j]: longest common sequence length up to X[i] and Y[j];
    flag[i][j]: path of LCS, (1,2,3) 1: jump diagonal, 2: jump down i-1 ->i, 3: jump right j-1 -> j 
  '''
  start_token = "START"
  X_new = [start_token] + list(X) # adding start token to X sequence
  Y_new = [start_token] + list(Y)
  
  m = len(X_new)
  n = len(Y_new)
  # starting length and flag matrix size : (m + 1) * (n + 1)
  length = [[0 for j in range(n)] for i in range(m)]
  flag = [[0 for j in range(n)] for i in range(m)]
  
  for i in range(1, m):
    for j in range(1, n):
      if (X_new[i] == Y_new[j]): # compare string
        length[i][j] = length[i-1][j-1] + 1
        flag[i][j] = 1 # diagonal
      else:
        if (length[i-1][j] > length[i][j-1]):
          length[i][j] = length[i-1][j]
          flag[i][j] = 2 # (i-1) -> i
        else:
          length[i][j] = length[i][j-1]
          flag[i][j] = 3 # (j-1) -> j
  return length, flag

#strA = "ABCBDAB"
#strB = "BDCABA" 
#m = LCS(strA, strB)

#listA = ['但是','我', '爱' ,'吃', '肉夹馍']
#listB = ['我', '不是', '很', '爱', '肉夹馍']
#m = LCS(listA, listB)


#y_candidate = ['我', '爱', '吃', '北京', '烤鸭']
#y_reference = [['我', '爱', '吃', '北京', '烧鸡'], ['我', '爱', '吃', '北京', '烤鹅'],['我', '很','爱', '吃', '西湖', '醋鱼']]
#p = eval_rouge_l(y_candidate, y_reference)

def plot_attention(data, X_label=None, Y_label=None):
  '''
    Plot the attention model heatmap
  '''
  fig, ax = plt.subplots()
  heatmap = ax.pcolor(data, cmap=plt.cm.Blues, alpha=0.8)
  
  ax.set_xticklabels(X_label, minor=False)
  ax.set_yticklabels(Y_label, minor=False)
  
  ax.grid(False)
  ax = plt.gca()
  # plt.show()
  fig.savefig('attention_weight.png')   # save the figure to file
  plt.close(fig)    # close the figure

def test():
  y_candidate = ['我', '爱', '吃', '北京', '烤鸭']
  y_reference = [['你', '爱', '吃', '北京', '小吃'], ['他', '爱', '吃', '北京', '烤鹅'],['但是', '我', '很','爱', '吃', '西湖', '醋鱼']]
  p1 = eval_rouge_l(y_candidate, y_reference)
  print (p1)
  
  p2 = eval_rouge_n(y_candidate, y_reference, 2)
  print (p2)

if __name__ == "__main__":
  test()
