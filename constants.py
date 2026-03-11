# -*- coding: utf-8 -*-

import math
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

# CPU clock frequency scales
KHZ = 1e3
MHZ = KHZ * 1e3
GHZ = MHZ * 1e3

KWh = 1000 * 3600

# CPU/GPU
GPU_capcity = 3e9  # 3080  FLOPS  核心数*频率*单时钟周期计算能力*精度转化系数--------默认单精度（fP32），整数(int8)，半精度（fp16），双精度（fp64）
CPU_capcity = 3e11  # i7 11代  FLOPS

CPU_frequency = 3.6 * GHZ


# Edge Network
es_num = 10
resource_type = 3
es_resource_type = [0, 1, 0, 1, 2, 0, 1, 2, 2, 2, 0, 0, 1, 1, 2, 0, 2, 1, 1, 0]  # 长度为es_num

# max_power_cpu = 8
# power_xpu_1 = 3.68  # GPU 2080ti sigle SM
# power_xpu_2 = 1  # DSP
# power_xpu_3 = 6  # FPGA
# max_power_xpu = [max_power_cpu, power_xpu_1, power_xpu_2]

model_type = [0, 1, 2, 3]
model_delay_matrix = np.array([[2.76, 1.58, 2.57], [6.45, 3.18, 3], [10.64, 7.78, 4.8], [33.9, 60.5, 21.6]], dtype=np.float32) / 1000
model_energy_matrix = np.array([[240, 14.57, 21.89], [570, 30.03, 30.9612], [930, 73.79, 67.3263], [5130, 34.42, 76.8868]], dtype=np.float32) / 1000


band_width = 20 * MB
ue_num_set = [2]

# ES
ES_xpu_frequency_set = [6e9]  # Hz


# DAG Tasks
alpha_set = [0.6, 0.8]
beta_set = [0.8]
data_size_set = [i * MB for i in np.arange(2, 4, 1)]

max_tole_delay_set = [1.0]
node_preference_set = [i for i in np.arange(0.1, 0.9, 0.1)]

coef = 1e-13  # the chip-dependent computing coefficient of UE


"""
arrival rate            Mbps
arrival data size       Mbps
time slot interval      sec (TBD)
Edge computation cap.   3.3*10^2~10^4
"""
