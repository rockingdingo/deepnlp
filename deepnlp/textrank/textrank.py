#!/usr/bin/python
# -*- coding:utf-8 -*-

"""Utilities for parsing textRank algorithm files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import math

def dtm(docs, min = 0):
  wordList = []
  for line in docs:
    for word in line:
      wordList.append(word)
  counter_word = collections.Counter(wordList)
  count_pairs_word = sorted(counter_word.items(), key=lambda x: (-x[1], x[0]))
  vocabList, _ = list(zip(*count_pairs_word))
  
  nDoc = len(docs)
  nVocab = len(vocabList)
  dtm = np.zeros((nDoc, nVocab))
  
  row_idx = 0
  col_idx = 0
  for line in docs:
    for word in line:
      idx = vocabList.index(word)
      dtm[row_idx, idx] += 1.0
    row_idx +=1
  return dtm, vocabList

def tfIdf(dtm):
  nDoc = dtm.shape[0]
  nTerm = dtm.shape[1]
  dtmNorm = dtm/dtm.sum(axis=1, keepdims=True) # Normalize tf to unit weight, tf/line word count
  dtmNorm = np.nan_to_num(dtmNorm)
  tfIdfMat = np.zeros((nDoc,nTerm))
  
  for j in range(nTerm):
    tfVect = dtmNorm[:, j]
    nExist = np.sum(tfVect > 0.0) # if tfVect is 0.0, word is not in current doc
    idf = 0.0
    # int32
    if (nExist > 0):
      idf = np.log(nDoc/nExist)/np.log(2) # log2()
    else:
      idf = 0.0
    tfIdfMat[:,j] = tfVect * idf
  
  return tfIdfMat

def cosine_similarity(v1,v2):
    # "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0.0, 0.0, 0.0
    sumxx = np.dot(v1, v1.T)
    sumyy = np.dot(v2, v2.T)
    sumxy = np.dot(v1, v2.T)
    cosine = 0.0
    if (math.sqrt(sumxx*sumyy) == 0.0):
        cosine = 0.0
    else:
        cosine = sumxy/math.sqrt(sumxx*sumyy)
    
    return cosine

# cosine_similarity(v1, v2)
# alpha = 0.01 threshold
def calcAdj(tfIdfMat):
  nDoc = tfIdfMat.shape[0]
  simMat = np.zeros((nDoc, nDoc))
  adjMat = np.zeros((nDoc, nDoc))
  for i in range(nDoc):
    for j in range(nDoc):
      v1 = tfIdfMat[i,:]
      v2 = tfIdfMat[j,:]
      cosine_sim = cosine_similarity(v1, v2)
      simMat[i, j] = cosine_sim
  
  # Use mean_sim instead of alpha, because tuning alpha is not applicable to all situations
  mean_sim = np.mean(simMat)
  alpha = mean_sim
  for i in range(nDoc):
    for j in range(nDoc):
      if (simMat[i,j] > alpha):
        adjMat[i, j] = 1.0
  return adjMat

def pagerank(nDim, adjMat, d, K):
    '''
    Args:
    d: damping factor, 
    K: iteration Number
    '''
    P = np.ones((nDim, 1)) * (1/nDim)
    
    # normalize adjacency Matrix
    B = adjMat/adjMat.sum(axis=1, keepdims=True)
    B = np.nan_to_num(B)
    
    U = np.ones((nDim, nDim)) * (1/nDim)
    
    M = d * B + (1-d) * U
    
    for i in range(K):
        P = np.dot(M.T, P)
    score = P.tolist()
    return P

def rank(docs, percent, order_by = 'id'):
    '''
    docs: input list of list, separated words;
    percent: percent of sentences to keep
    '''
    
    # Default Parameters
    # alpha = 0.1
    d = 0.1 
    K = 500
    
    nDoc = len(docs)
    docTermMat, vocab = dtm(docs)
    tfIdfMat = tfIdf(docTermMat)
    adjMat = calcAdj(tfIdfMat)
    score = pagerank(nDoc, adjMat, d, K)
    
    docScore = []
    for i in range(nDoc):
        docScore.append((i, docs[i], score[i])) # return list of tuples [(index, doc, score)]
    
    # Ranking
    sortedList = sorted(docScore, key=lambda item : -item[2]) # sort to desc order of score
    nKeep = np.int(nDoc * percent)
    doc_sum = sortedList[0: nKeep] # doc_sum: nKeep number of document highest score
    
    if (order_by == 'score') :
        rank_idx = 2
    elif (order_by == 'id'):
        rank_idx = 0
    else :
        rank_idx = 0
    doc_sum_rerank = sorted(doc_sum, key=lambda item : item[rank_idx]) # nKeep number of document sort to increase order of index
    return doc_sum_rerank

if __name__ == '__main__':
  
  # document term Matrix
  docs = [['今天','天气','很好', '看'],['天气', '很好'],['今天'],['你是', '谁', '谁']]
  percent = 0.5
  list = rank(docs, percent)
  print (list)

