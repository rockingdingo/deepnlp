#!/usr/bin/python
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals # compatible with python3 unicode

### Define Constant Variables for Scope Name
_pos_scope_name="pos_var_scope"

_pos_variables_namescope="pos_variables"

_ner_scope_name="ner_var_scope"

_ner_variables_namescope="ner_variables"

_parse_scope_name="parse_var_scope"

_parse_variables_namescope="parse_variables"

### Define Registered Model Name that can be downloaded
registered_models = [
    {
        'segment': [
                'zh', 
                'zh_o2o', 
                'zh_entertainment'
            ],
        'ner': [
                'zh', 
                'zh_o2o', 
                'zh_entertainment'
            ],
        'pos': [
                'zh', 
                'en'
            ],
        'parse': [
                'zh'
            ],
    },
]

def get_model_var_scope(module_scope, model_name):
  """ Assembly Module Scope Name and model scope name
  """
  model_var_scope = module_scope + "_" + model_name
  return model_var_scope

def main():
    return

if __name__ == '__main__':
    main()
