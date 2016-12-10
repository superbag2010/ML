#! /usr/bin/python3

import sys 
import os
import re

sum_RMSE = 0
in_dir = sys.argv[1]
len_lines = 0 
RMSE_result_path = os.path.abspath(os.path.join(in_dir, "RMSE_result.txt"))
with open(RMSE_result_path, 'r') as f:
    lines = f.readlines()
    len_lines = len(lines)
    for line in lines:
        RMSE = line.split(',')
        RMSE[1] = re.sub(r"\n", "", RMSE[1])
        sum_RMSE += float(RMSE[1])

with open(RMSE_result_path, 'a') as f:
    f.write("Mean of RMSE(the number of RMSE={}) : ".format(len_lines) + str(sum_RMSE/len_lines) + "\n")
