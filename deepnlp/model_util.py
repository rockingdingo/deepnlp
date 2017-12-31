#!/usr/bin/python
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals # compatible with python3 unicode

import codecs

### Define Constant Variables for Scope Name
_pos_scope_name="pos_var_scope"

_pos_variables_namescope="pos_variables"

_ner_scope_name="ner_var_scope"

_ner_variables_namescope="ner_variables"

_parse_scope_name="parse_var_scope"

_parse_variables_namescope="parse_variables"

### Define Registered Model Name that can be downloaded and registered
global registered_models
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

#### register the usered_defined model before use
def register_model(module, name):
    """ Brief: Register the model name under module
    """
    global registered_models
    if module in registered_models[0]:
        print ("NOTICE: Start Registering model %s under module %s" % (name, module))
        registered_models[0][module].append(name)
        print ("NOTICE: All registered modules are %s" % str(registered_models[0][module]))
    else:
        print ("DEBUG: Registered module name %s is not allowed..." % module)

def get_model_var_scope(module_scope, model_name):
  """ Assembly Module Scope Name and model scope name
  """
  model_var_scope = module_scope + "_" + model_name
  return model_var_scope

#### Model Config Util
def load_config(filepath):
  """ Load Conf from filepath
  """
  file = codecs.open(filepath, encoding='utf-8')
  lines = []
  for line in file:
    line = line.strip()
    lines.append(line)
  model_config_dict = {}            # config map for all models
  model_name = None
  model_config = {}                 # config for one model
  for line in lines:
    line = line.strip()
    if (line.startswith("[")):      # [model_config_name=zh]
      model_config = {}             # create new empty config dict
      model_config_names = line.replace("[","").replace("]","").split("=")
      if (len(model_config_names) == 2):
        model_name = model_config_names[1]
    elif (line.startswith("#")):    # comments
      continue
    elif (line.strip() == ""):      # Blank line, Previous Config is finished, Adding current config
      if (len(model_config) > 0 and model_name is not None):
        model_config_dict[model_name] = model_config
    else:
      items = line.split("=")
      if (len(items) == 2):
        key = items[0].strip()
        value = items[1].strip()
        model_config[key]= value
  # Add last config
  if (len(model_config) > 0 and model_name is not None):
    model_config_dict[model_name] = model_config
  # find results
  print ("NOTICE: LOADING Model Config number %d ..." % len(model_config_dict))
  return model_config_dict

_default_model_name = "default"

class DefaultModelConfig(object):
  def __init__(self):
    self.name = _default_model_name

#### Get Model Config
def get_config(conf_dict, name):
  """ Brief:get model from conf_dict
      Args: conf_dict:  2Dmap <K,V> K:model_name, V: dict{} hyper_param_key=hyper_param_value
  """
  if (name not in conf_dict):
    print ("DEBUG: Model Name not in models.conf file... %s" % name)
    print ("DEBUG: Loading default model config")
    config = DefaultModelConfig()
    if _default_model_name in conf_dict:
      default_param = conf_dict[_default_model_name]
      for key in default_param.keys():
        setattr(config, key, eval(default_param[key]))
    return config
  else:
    model_param = conf_dict[name]
    config = DefaultModelConfig()
    try:
      for key in model_param.keys():
        setattr(config, key, eval(model_param[key]))    # convert hparam from string to int/float
    except Exception as e:
      print (e)
    return config

def main():
    return

if __name__ == '__main__':
    main()
