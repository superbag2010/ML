# ML
machine learning project
- predict the number of cold treatment using evrionment data(temperature, etc)


Before train and evaluation, must check following flag.
1. window_height,               e.g. 7
2. out_subdir,                  e.g. "num_NN_nodes"
3. factor_value                 e.g. "60,45,30"
-evaluation-
4. checkpoint
5. eval_train

-training- e.g.
./train.py --window_height=7 --out_subdir="num_NN_nodes" --factor_value="60,45,30"
(result is saved in "../../result_disease_cnn/num_NN_nodes/60,45,30/<datetime>")

-evaluation- e.g.
./eval.py --window_height=7 --checkpoint_dir="../../result_disease_cnn/num_NN_nodes/60,45,30/<datetime>/checkpoint" --eval_train="True"

