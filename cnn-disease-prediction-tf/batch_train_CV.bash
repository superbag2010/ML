#! /bin/bash

# used for training at once, and Cross validation

#./train.py --window_height=5 --out_subdir="sigmoid/num_NN_nodes" --num_nodes="0" --factor_value="0"
#./train.py --window_height=5 --out_subdir="sigmoid/num_NN_nodes" --num_nodes="45" --factor_value="45"
#./train.py --window_height=5 --out_subdir="sigmoid/num_NN_nodes" --num_nodes="100,45" --factor_value="100,45"
#./train.py --window_height=5 --out_subdir="sigmoid/num_NN_nodes" --num_nodes="140,70,25" --factor_value="140,70,25"

#./train.py --window_height=6 --out_subdir="sigmoid/num_NN_nodes/0/window_height" --num_nodes="0" --factor_value="6" --num_epochs=500 --filter_sizes="1,2,3,4,5,6"
#./train.py --window_height=5 --out_subdir="sigmoid/num_NN_nodes/0/window_height" --num_nodes="0" --factor_value="5" --num_epochs=500 --filter_sizes="1,2,3,4,5"
#./train.py --window_height=4 --out_subdir="sigmoid/num_NN_nodes/0/window_height" --num_nodes="0" --factor_value="4" --num_epochs=500 --filter_sizes="1,2,3,4"
#./train.py --window_height=3 --out_subdir="sigmoid/num_NN_nodes/0/window_height" --num_nodes="0" --factor_value="3" --num_epochs=500 --filter_sizes="1,2,3"
#./train.py --window_height=2 --out_subdir="sigmoid/num_NN_nodes/0/window_height" --num_nodes="0" --factor_value="2" --num_epochs=500 --filter_sizes="1,2"


./train.py --window_height=2 --out_subdir="sigmoid/num_NN_nodes/250/" --num_nodes="250" --filter_sizes="1,2"
./train.py --window_height=2 --out_subdir="sigmoid/num_NN_nodes/250/" --num_nodes="250" --filter_sizes="1,2"
./mean_RMSE.py "../../result_disease_cnn/sigmoid/num_NN_nodes/250/"


./train.py --window_height=2 --out_subdir="sigmoid/num_NN_nodes/150/" --num_nodes="150" --filter_sizes="1,2"
./train.py --window_height=2 --out_subdir="sigmoid/num_NN_nodes/150/" --num_nodes="150" --filter_sizes="1,2"
./mean_RMSE.py "../../result_disease_cnn/sigmoid/num_NN_nodes/150/"


./train.py --window_height=2 --out_subdir="sigmoid/num_NN_nodes/60/" --num_nodes="60" --filter_sizes="1,2"
./train.py --window_height=2 --out_subdir="sigmoid/num_NN_nodes/60/" --num_nodes="60" --filter_sizes="1,2"
./mean_RMSE.py "../../result_disease_cnn/sigmoid/num_NN_nodes/60/"


./train.py --window_height=2 --out_subdir="sigmoid/num_NN_nodes/320,30/" --num_nodes="320,30" --filter_sizes="1,2"
./train.py --window_height=2 --out_subdir="sigmoid/num_NN_nodes/320,30/" --num_nodes="320,30" --filter_sizes="1,2"
./mean_RMSE.py "../../result_disease_cnn/sigmoid/num_NN_nodes/320,30/"


./train.py --window_height=2 --out_subdir="sigmoid/num_NN_nodes/280,50/" --num_nodes="280,50" --filter_sizes="1,2"
./train.py --window_height=2 --out_subdir="sigmoid/num_NN_nodes/280,50/" --num_nodes="280,50" --filter_sizes="1,2"
./mean_RMSE.py "../../result_disease_cnn/sigmoid/num_NN_nodes/280,50/"


./train.py --window_height=2 --out_subdir="sigmoid/num_NN_nodes/220,100/" --num_nodes="220,100" --filter_sizes="1,2"
./train.py --window_height=2 --out_subdir="sigmoid/num_NN_nodes/220,100/" --num_nodes="220,100" --filter_sizes="1,2"
./mean_RMSE.py "../../result_disease_cnn/sigmoid/num_NN_nodes/220,100/"
