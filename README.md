# ML
machine learning project
- predict the number of cold treatment using evrionment data(temperature, etc)


Before train and evaluation, must check following flag.
1. window_height
2. out_subdir
3. factor_value

./train.py --out_subdir="<hyper parameter name>" --factor_value="<factor value>"
./eval.py --checkpoint_dir="../../result_disease_cnn/num_NN_layer/<number>/checkpoints/" --eval_train="True"

