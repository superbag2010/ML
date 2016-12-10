#! /usr/bin/python3

import datetime
import time
import os

a = str(1)
timestamp = str(int(time.time()))
RMSE_result_path = os.path.join("./","RMSE_result.txt")
with open(RMSE_result_path, 'a') as f:
    f.write(timestamp + "," + a + "\n")
