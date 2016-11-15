#!/bin/bash

# Train pos model adding short language option code:
# python pos_model.py arg1
# arg1 takes language code as input 'en' for english, 'zh' for Chinese
# data and model folder will be created under .../pos/data/en and .../pos/ckpt/en

python pos_model.py en # LSTM model English

python pos_model.py zh # LSTM model Chinese

python pos_model_bilstm.py en # Bi-LSTM model English

python pos_model_bilstm.py zh # Bi-LSTM model Chinese

