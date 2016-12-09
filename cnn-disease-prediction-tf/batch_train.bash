#! /bin/bash

#./train.py --window_height=5 --out_subdir="sigmoid/num_NN_nodes" --num_nodes="0" --factor_value="0"
#./train.py --window_height=5 --out_subdir="sigmoid/num_NN_nodes" --num_nodes="45" --factor_value="45"
#./train.py --window_height=5 --out_subdir="sigmoid/num_NN_nodes" --num_nodes="100,45" --factor_value="100,45"
#./train.py --window_height=5 --out_subdir="sigmoid/num_NN_nodes" --num_nodes="140,70,25" --factor_value="140,70,25"

./train.py --window_height=6 --out_subdir="sigmoid/num_NN_nodes/0/window_height" --num_nodes="0" --factor_value="6" --num_epochs=500 --filter_sizes="1,2,3,4,5,6"
./train.py --window_height=5 --out_subdir="sigmoid/num_NN_nodes/0/window_height" --num_nodes="0" --factor_value="5" --num_epochs=500 --filter_sizes="1,2,3,4,5"
./train.py --window_height=4 --out_subdir="sigmoid/num_NN_nodes/0/window_height" --num_nodes="0" --factor_value="4" --num_epochs=500 --filter_sizes="1,2,3,4"
./train.py --window_height=3 --out_subdir="sigmoid/num_NN_nodes/0/window_height" --num_nodes="0" --factor_value="3" --num_epochs=500 --filter_sizes="1,2,3"
./train.py --window_height=2 --out_subdir="sigmoid/num_NN_nodes/0/window_height" --num_nodes="0" --factor_value="2" --num_epochs=500 --filter_sizes="1,2"
