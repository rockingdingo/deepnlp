#!/bin/bash

# output_node包含var_scope/node_name

# freeze the graph and the weights
python freeze_graph.py --input_graph=./ner_graph.pbtxt --input_checkpoint=../ckpt/zh_o2o/ner.ckpt --output_graph=./ner_graph_frozen.pb --output_node_names=ner_var_scope/output_node

