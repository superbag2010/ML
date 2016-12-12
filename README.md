# ML(version of tensorflow = 0.11.0rc1)

### machine learning project
predict the number of cold treatment using evrionment data(temperature, etc)  
 \- Can change variety flag and check in "<train_result>/flag.conf" file

================================================================
### Before train and evaluation, "must check" following flag.  
1. window_height   &nbsp        e.g. 7  
2. out_subdir(directory name)   e.g. "num_NN_nodes"  
3. filter_size                  e.g. "1,2,3"
-evaluation-  
4. data_file_location
5. checkpoint  

================================================================
### Hyper Prameter  
1. Data format  
\- window_height  
\- num_features  

2. hidden layer 
\- num_nodes            e.g. "60,45,30"  
\- ("0" mean no hidden layer)  

3. filter  
\- filter_sizes         e.g. "3,4,5,6,7"  
\- num_filters          e.g. 128  

4. regularization  
\- dropout_keep_prob    e.g. 0.8  
\- l2_reg_lambda        e.g. 0.0  

5. training interval  
\- batch_size           e.g. 1  
\- num_epochs          e.g. 200  

  etc.. planning to add  
  activation function, learning rate  

================================================================
### EXAMPLE
\- training- e.g.  
./train.py --window_height=7 --out_subdir="num_NN_nodes/64,45,30" --num_nodes="60,45,30"
(result is saved in "../../result_disease_cnn/num_NN_nodes/60,45,30/<datetime>")  

\- evaluation- e.g.  
./eval.py --window_height=7 --checkpoint_dir="../../result_disease_cnn/num_NN_nodes/60,45,30/<datetime>/checkpoint" --data_file_location="./data/refined_data.csv"

\- training at once and cross validation- e.g.
./mean_RMSE.py "../../result_disease_cnn/sigmoid/num_NN_nodes/0/window_height/1"

\- tensorboard --logdir="../../result_disease_cnn/refined_data/0/1481498168/summaries/"
