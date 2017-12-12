#!/usr/bin/python
# -*- coding:utf-8 -*-
""" 
Beam Search module for Parser
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals # compatible with python3 unicode coding

import time
import numpy as np
import tensorflow as tf
import sys,os

pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # .../deepnlp/.
sys.path.append(pkg_path)

from parse import transition_system    # absolute import
from parse import reader
from parse.transition_system import Configuration

class Hypothesis(object):
    tokens = []     # sequence of tokens of current hyps
    score = 0.0     #evaluation score of current hyps

    def __init__(self, tokens, score):
        self.tokens = tokens
        self.score = score

    def extend(self, token, score):
        """ Return a new Hpys based on last step with new score
        """
        return Hypothesis(self.tokens + token, self.score + score)

    @property
    def latest_token(self):
        return self.tokens[-1] if len(tokens) > 0 else None


class BeamSearch(object):
    """ Maintain a Beam size of K with the highest scored Hypothesis
    """
    def __init__(self, beam_size):
        self._beam_size = beam_size
        self._hyps = [Hypothesis([], 0.0)] * self._beam_size   # a list of empty hypothesis

    def search(self, sess, model, feature):
        """ Return the highest scored top K hyps
        """
        results = []
        latest_tokens = [h.latest_token for h in self.hyps]   # last_tokens of previous step
        top_ids, top_scores = model.decode_topK(sess, feature)

        # Extend each hypothesis.
        all_hyps = []
        for i in range(self._beam_size):
            #for the ith hyps, extend it sequence
            h = self._hyps[i]
            for j in range(len(top_ids)):
                all_hyps.append(h.extend(top_ids[j], top_scores[j]))   # append new Hyps
        # Get topK hyps
        return self.get_topk(all_hyps)
    
    def get_topk(self, hyps):
        """Sort the hyps based on score.
        Args:
          hyps: A list of hypothesis.
        Returns:
          hyps: A list of sorted hypothesis in mininum global score (negative log likelihood), achieve max likelihood
        """
        return sorted(hyps, key=lambda h: h.score, reverse=False)  # increase order
    
    @property
    def hyps(self):
        return self._hyps
