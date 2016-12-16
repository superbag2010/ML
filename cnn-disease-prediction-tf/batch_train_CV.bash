#! /bin/bash

# used for training at once, and Cross validation

#./train.py --window_height=5 --out_subdir="sigmoid/num_NN_nodes" --num_nodes="0" --factor_value="0"
#./train.py --window_height=5 --out_subdir="sigmoid/num_NN_nodes" --num_nodes="45" --factor_value="45"
#./train.py --window_height=5 --out_subdir="sigmoid/num_NN_nodes" --num_nodes="100,45" --factor_value="100,45"
#./train.py --window_height=5 --out_subdir="sigmoid/num_NN_nodes" --num_nodes="140,70,25" --factor_value="140,70,25"

#./train.py --window_height=6 --out_subdir="sigmoid/num_NN_nodes/0/window_height" --num_nodes="0" --factor_value="6" --num_epochs=500 --filter_sizes="1,2,3,4,5,6"
#./train.py --window_height=3 --out_subdir="sigmoid/num_NN_nodes/0/window_height" --num_nodes="0" --factor_value="3" --num_epochs=500 --filter_sizes="1,2,3"
#./train.py --window_height=2 --out_subdir="sigmoid/num_NN_nodes/0/window_height" --num_nodes="0" --factor_value="2" --num_epochs=500 --filter_sizes="1,2"



./train.py --window_height=1 --out_subdir="cold_data/0/height1/" --num_nodes="0" --filter_sizes="1" --data_file_location="./data/cold_data.csv"

./train.py --window_height=2 --out_subdir="cold_data/0/height2/" --num_nodes="0" --filter_sizes="1,2" --data_file_location="./data/cold_data.csv"

./train.py --window_height=3 --out_subdir="cold_data/0/height3/" --num_nodes="0" --filter_sizes="1,2,3" --data_file_location="./data/cold_data.csv"

./train.py --window_height=2 --out_subdir="cold_data/250/" --num_nodes="250" --filter_sizes="1,2" --data_file_location="./data/cold_data.csv"

./train.py --window_height=2 --out_subdir="cold_data/280,50/" --num_nodes="280,50" --filter_sizes="1,2" --data_file_location="./data/cold_data.csv"

./train.py --window_height=2 --out_subdir="cold_data/290,130,30/" --num_nodes="280,50" --filter_sizes="1,2" --data_file_location="./data/cold_data.csv"


./train.py --window_height=1 --out_subdir="cold_data/0/height1/" --num_nodes="0" --filter_sizes="1" --data_file_location="./data/cold_data.csv"

./train.py --window_height=2 --out_subdir="cold_data/0/height2/" --num_nodes="0" --filter_sizes="1,2" --data_file_location="./data/cold_data.csv"

./train.py --window_height=3 --out_subdir="cold_data/0/height3/" --num_nodes="0" --filter_sizes="1,2,3" --data_file_location="./data/cold_data.csv"

./train.py --window_height=2 --out_subdir="cold_data/250/" --num_nodes="250" --filter_sizes="1,2" --data_file_location="./data/cold_data.csv"

./train.py --window_height=2 --out_subdir="cold_data/280,50/" --num_nodes="280,50" --filter_sizes="1,2" --data_file_location="./data/cold_data.csv"

./train.py --window_height=2 --out_subdir="cold_data/290,130,30/" --num_nodes="280,50" --filter_sizes="1,2" --data_file_location="./data/cold_data.csv"



./mean_RMSE.py "../../result_disease_cnn/cold_data/0/height1"
./mean_RMSE.py "../../result_disease_cnn/cold_data/0/height2"
./mean_RMSE.py "../../result_disease_cnn/cold_data/0/height3"
./mean_RMSE.py "../../result_disease_cnn/cold_data/250/"
./mean_RMSE.py "../../result_disease_cnn/cold_data/280,50/"
./mean_RMSE.py "../../result_disease_cnn/cold_data/290,130,30"

