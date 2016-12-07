# ML(version of tensorflow = 0.11.0rc1)
machine learning project
- predict the number of cold treatment using evrionment data(temperature, etc)

Can change variety flag and check in "<train_result>/flag.conf" file

================================================================
Before train and evaluation, "must check" following flag.
1. window_height                e.g. 7
2. out_subdir(directory name)   e.g. "num_NN_nodes"
3. factor_value(directory name) e.g. "60,45,30"
4. num_features
-evaluation-
4. checkpoint
5. eval_train

================================================================
Hyper Prameter
- Data format - 
1. window_height
2. num_features

- hidden layer -
1. num_nodes            e.g. "60,45,30"
("0" mean no hidden layer)

- filter -
1. filter_sizes         e.g. "3,4,5,6,7"
2. num_filters          e.g. 128

- regularization -
1. dropout_keep_prob    e.g. 0.8
2. l2_reg_lambda        e.g. 0.0

- training interval -
1. batch_size           e.g. 1
2. num_epoches          e.g. 200

etc.. planning to add
activation function, learning rate
================================================================
-training- e.g.
./train.py --window_height=7 --out_subdir="num_NN_nodes" --num_nodes="60,45,30" --factor_value="60,45,30"
(result is saved in "../../result_disease_cnn/num_NN_nodes/60,45,30/<datetime>")

-evaluation- e.g.
./eval.py --window_height=7 --checkpoint_dir="../../result_disease_cnn/num_NN_nodes/60,45,30/<datetime>/checkpoint" --eval_train="True"

