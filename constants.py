# -*- coding: utf-8 -*-


import numpy as np
import os

# 获取当前文件所在目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Data size scales
BYTE = 8
KB = 1024 * BYTE
MB = 1024 * KB
GB = 1024 * MB
TB = 1024 * GB
PB = 1024 * TB


ms = 0.001


# Edge Network

resource_type = 3
es_resource_type = [0, 1, 0, 1, 2, 0, 1, 2, 2, 2, 0, 0, 1, 1, 2, 0, 2, 1, 1, 0]  # 长度为es_num


model_type = [0, 1, 2, 3]
model_delay_matrix = np.array([[2.76, 1.58, 2.57], [6.45, 3.18, 3], [10.64, 7.78, 4.8], [33.9, 60.5, 21.6]], dtype=np.float32) / 1000
model_energy_matrix = np.array([[240, 17.2, 21.89], [570, 36.57, 30.96], [930, 88.13, 67.3263], [3868.2, 850.82, 76.89]], dtype=np.float32) / 1000


band_width = 20 * MB
ue_num_set = [2]


# DAG Tasks

data_size_set = [i * MB for i in np.arange(2, 4, 1)]


"""
arrival rate            Mbps
arrival data size       Mbps
time slot interval      sec (TBD)
Edge computation cap.   3.3*10^2~10^4
"""
