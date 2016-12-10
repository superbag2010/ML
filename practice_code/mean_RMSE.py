#! /usr/bin/python3

import sys
import os

sum_RMSE = 0
in_dir = sys.argv[1]
len_lines = 0
RMSE_result_path = os.path.abspath(os.path.join(in_dir, "RMSE_result.txt"))
with open(RMSE_result_path, 'r') as f:
    lines = f.readlines()
    len_lines = len(lines)
    for line in lines:
        RMSE = line.split(',')
        sum_RMSE += int(RMSE[1])

with open(RMSE_result_path, 'a') as f:
    f.write("Mean of RMSE(the number of RMSE={}) : ".format(len_lines) + str(sum_RMSE/len_lines))
