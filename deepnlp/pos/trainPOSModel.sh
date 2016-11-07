#!/bin/bash

# Train pos model adding short lang option code:
# python pos_model.py arg1
# arg1 takes language code as input 'en' for english, 'zh' for Chinese
# data and model folder will be created under .../pos/data/en and .../pos/ckpt/en

python pos_model.py en

python pos_model.py zh

