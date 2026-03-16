import numpy as np
import os
import math

import constants as cn
import pandas as pd
import os
import glob


def number_to_onehot(num, max_val, min_val):
    """
    将自然数转换为one-hot编码

    参数：
    num     : 要转换的自然数（整数）
    max_val : 允许的最大值（包含）
    min_val : 允许的最小值（默认0，包含）

    返回：
    list类型，长度为 max_val - min_val + 1

    示例：
    >>> number_to_onehot(0, 2)
    [1, 0, 0]
    >>> number_to_onehot(1, 2)
    [0, 1, 0]
    """
    # 参数校验
    if not isinstance(num, int):
        raise TypeError(f"输入必须是整数,当前为{type(num)}")
    if not (min_val <= num <= max_val):
        raise ValueError(f"数值必须在 [{min_val}, {max_val}] 范围内")
    if max_val < min_val:
        raise ValueError("最大值必须大于等于最小值")

    # 计算类别总数
    num_classes = max_val - min_val + 1

    # 生成全零向量
    onehot = [0] * num_classes

    # 设置对应位置为1
    onehot[num - min_val] = 1

    return onehot


# 进制转换
def ten2(n, x, k):
    """
    输入（十进制数,目标进制数<99,输出位数）
    输出的数值为列表形式，每个数字之间用逗号隔开
    """
    # n为待转换的十进制数，x为机制，取值为2-16
    a = [i for i in range(100)]
    b = []
    while True:
        s = n // x  # 商
        y = n % x  # 余数
        b = b + [y]
        if s == 0:
            break
        n = s
    b.reverse()
    temp = [a[i] for i in b]
    while True:
        if len(temp) < k:
            temp.insert(0, 0)
        else:
            break
    return temp
