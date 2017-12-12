#!/bin/bash

# freeze the graph and the weights
python freeze_graph.py --input_graph=../model/nn_model.pbtxt --input_checkpoint=../ckpt/nn_model.ckpt --output_graph=../model/nn_model_frozen.pb --output_node_names=output_node

