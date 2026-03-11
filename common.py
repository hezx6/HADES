import numpy as np
import os
import math
import csv
import constants as cn
import pandas as pd
import os
import glob


def delete_files_with_prefix(target_dir, prefix):

    # 构造匹配模式（匹配目录下所有以xmd开头的文件）
    pattern = os.path.join(target_dir, f"{prefix}*")

    # 安全检查：先列出匹配的文件
    matched_files = glob.glob(pattern)
    print("以下文件将被删除：")
    for file in matched_files:
        print(f" - {file}")

    # 执行删除
    try:
        for file in matched_files:
            if os.path.isfile(file):  # 确保是文件而非目录
                os.remove(file)
                print(f"已删除: {file}")
        print("操作完成")
    except Exception as e:
        print(f"发生错误: {str(e)}")


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


def convert_(tar, list):
    """
    将tar转换成list中的的对应进制(混合进制转化),list最后两位必须一样
    """
    result = [0, 0, 0, 0]
    theread = [list[3] * list[2] * list[1] * list[0], list[1] * list[2] * list[3], list[2] * list[3], list[3]]

    if tar <= theread[2]:
        temp = ten2(tar, list[2], 2)
        result[2], result[3] = temp[0], temp[1]
    elif tar <= theread[1] and tar > theread[2]:
        result[1] = tar // theread[2]
        tar = tar - result[1] * theread[2]
        temp = ten2(tar, list[2], 2)
        result[2], result[3] = temp[0], temp[1]
    elif tar <= theread[0] and tar > theread[1]:
        result[0] = tar // theread[1]
        tar = tar - result[0] * theread[1]
        result[1] = tar // theread[2]
        tar = tar - result[1] * theread[2]
        temp = ten2(tar, list[2], 2)
        result[2], result[3] = temp[0], temp[1]
    return result


def read_bandwidth_data(file_path, m, n):
    """
    返回给定时间点之前的一定长度的值序列(不包括给定的时间点的数据值)
    如(10,8)返回第2~9共8个值组成的列表
    """
    with open(file_path, "r") as file:
        lines = file.readlines()
        # 读取指定索引范围的数据
        bandwidth_list = [float(line.split()[4]) for line in lines[m - n - 1 : m - 1]]
    return bandwidth_list


def query_key(lines, key):

    # 遍历文件内容,除了标题行
    for parts in lines[1:]:
        # 解析每一行的数据
        if (int(parts[0]), parts[1], int(parts[2]), int(parts[3])) == key:
            return int(parts[4])
    raise KeyError(f"Key not found: {key}")


def read_nth_line_mth_column(lines, n, m):

    n = n % 500
    if n > len(lines) or m > len(lines[n - 1].split()):
        print("Error when reading the network speed!")
        return None
    return int(lines[n - 1].split()[m - 1])


def ran_int(Range):
    """
    随机在给定范围内的挑出一个整数
    """
    temp = np.random.uniform(0, 1)
    return Range[0] + round(temp * (Range[1] - Range[0]))


def ran_float(Range):
    """
    随机在给定范围内的挑出一个小数,保留3位小数
    """
    temp = np.random.uniform(0, 1)
    return int((Range[0] + temp * (Range[1] - Range[0])) * 1000) / 1000


def ran01():
    return np.random.uniform(0, 1)


def yes_no(prob):
    if 0 <= prob <= 1:
        return ran01() < prob
    else:
        return None


# for i in range(10):
#      print(ran_float([0.7,0.8]))
