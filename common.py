import numpy as np
import os
import math
import csv
import constants as cn
import pandas as pd
import os
import glob

# ================= 设置自定义 matplotlibrc 路径  ====================
import matplotlib as mpl

custom_matplotlibrc_path = "/home/hzx/Plot/MyMatplotlibrc"
os.environ["MATPLOTLIBRC"] = custom_matplotlibrc_path  # 必须在导入 pyplot 前设置
print("当前使用的配置文件路径:", mpl.matplotlib_fname())

# mpl.rc_file_defaults()  # 重置为默认配置（可选）
mpl.rc_file(custom_matplotlibrc_path)  # 加载自定义配置!  重要步骤

# 之后导入 pyplot
import matplotlib.pyplot as plt

# ==================================================================

font_config = {"size": 15, "weight": "normal", "family": "serif"}  # weight数值100~900（数字越大越粗）


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


def plot_CDF_and_PMF(data, save_dir):
    """
    绘制给定列表的累积分布函数(CDF)。

    参数:
    data (list): 输入数据列表，需为数值类型。
    save_dir:图片保存的位置

    返回:
    None (直接显示图像)。

    """
    if not data:
        raise ValueError("输入列表不能为空。")

    # 处理数据并计算CDF
    sorted_data = np.sort(data)
    unique_values, counts = np.unique(sorted_data, return_counts=True)
    cumulative_prob = np.cumsum(counts) / len(sorted_data)

    # 绘制阶梯图
    # 创建双子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2))  # 双图横向排列
    plt.subplots_adjust(wspace=0.3)  # 调整子图间距

    # CDF子图
    ax1.step(unique_values, cumulative_prob, where="post", color="#9F0000", linewidth=2)  # ["#9F0000", "#003A75", "#FD5F5E", "AFD9FD"]
    ax1.set_xlabel("Latency", fontdict=font_config)
    ax1.set_ylabel("CDF", fontdict=font_config)
    ax1.grid(True, linestyle="-.", alpha=0.7)
    ax1.set_xlim(unique_values[0] - 0.1, unique_values[-1] + 0.1)

    # PMF子图（概率-时延图）
    # 计算分箱参数
    min_val = np.min(data)
    max_val = np.max(data)
    num_bins = 10

    # 生成等宽区间边界（扩展0.1%防止边界值溢出）
    bin_edges = np.linspace(min_val - 0.001 * (max_val - min_val), max_val + 0.001 * (max_val - min_val), num_bins + 1)

    # 计算区间统计
    hist_counts, _ = np.histogram(data, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # 使用区间中点作为x坐标
    bin_width = bin_edges[1] - bin_edges[0]
    probabilities = hist_counts / len(data)

    ax2.plot(bin_centers, probabilities, color="#003A75", linewidth=2, marker="v")
    ax2.set_xlabel("Latency", fontdict=font_config)
    ax2.set_ylabel("Probability", fontdict=font_config)
    ax2.set_xlim(bin_edges[0] - bin_width, bin_edges[-1] + bin_width)
    ax2.grid(True, linestyle="-.", alpha=0.7)

    # 其他设置
    plt.title(None)
    plt.tight_layout()
    plt.savefig(save_dir, dpi=200)
    plt.close()


def write_data_to_csv(file_path, data, mode):
    """
    用 pandas 实现的 CSV 写入函数（支持三种模式）

    :param file_path: 文件路径
    :param data: 数据（一维列表）
    :param mode:
        - 'overwrite': 覆盖写入（清空文件后写入）
        - 'append': 追加写入（保留历史数据）
        - 'overwrite_last': 覆盖最后一行（保留其他数据）
    """
    # 创建目录
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # 将数据转换为 DataFrame
    df_new = pd.DataFrame([data])

    # 处理不同写入模式
    if mode == "overwrite":
        # 覆盖写入
        df_new.to_csv(file_path, index=False, header=False)
    elif mode == "append":
        # 追加模式
        if not os.path.exists(file_path):
            # 文件不存在时直接覆盖写入（等同于首次追加）
            df_new.to_csv(file_path, index=False, header=False)
        else:
            # 行追加：直接追加到文件末尾
            df_new.to_csv(file_path, mode="a", index=False, header=False)
    elif mode == "overwrite_last":
        # 覆盖最后一行：保留其他数据
        if not os.path.exists(file_path):
            # 文件不存在时直接写入
            df_new.to_csv(file_path, index=False, header=False)
        else:
            # 读取现有数据并删除最后一行
            df = pd.read_csv(file_path)
            if not df.empty:
                df = df.iloc[:-1]  # 移除最后一行
            # 追加新数据
            df = pd.concat([df, df_new], ignore_index=True)
            df.to_csv(file_path, index=False, header=False)
    else:
        raise NotImplementedError(f"Unsupported mode: {mode}. Supported modes are 'overwrite', 'append', 'overwrite_last'.")


def write_data_to_csv_new(file_path, data, mode):
    """
    优化后的CSV写入函数（支持三种模式）

    :param file_path: 文件路径
    :param data: 数据（一维列表）
    :param mode:
        - 'overwrite': 覆盖写入（清空文件后写入）
        - 'append': 追加写入（保留历史数据）
        - 'overwrite_last': 覆盖最后一行（保留其他数据）
    """
    # 创建目录（优化：先检查再创建）
    dir_path = os.path.dirname(file_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    file_exists = os.path.exists(file_path)

    if mode == "overwrite":
        with open(file_path, "w", newline="") as f:
            csv.writer(f).writerow(data)

    elif mode == "append":
        with open(file_path, "a" if file_exists else "w", newline="") as f:
            csv.writer(f).writerow(data)

    elif mode == "overwrite_last":
        if not file_exists:
            with open(file_path, "w", newline="") as f:
                csv.writer(f).writerow(data)
        else:
            # 避免读取整个文件：直接定位最后一行
            temp_file = file_path + ".tmp"
            with open(file_path, "r", newline="") as fin, open(temp_file, "w", newline="") as fout:
                reader = csv.reader(fin)
                writer = csv.writer(fout)

                # 复制除最后一行的所有数据
                prev_line = None
                for current_line in reader:
                    if prev_line is not None:
                        writer.writerow(prev_line)
                    prev_line = current_line

                # 写入新数据作为最后一行
                writer.writerow(data)

            os.replace(temp_file, file_path)


def get_next_log_file_number(log_dir):
    max_log_number = 0
    for filename in os.listdir(log_dir):
        if filename.startswith("run_") and filename.endswith(".log"):
            log_number = int(filename[len("run_") : len("run_") + 1])
            if log_number > max_log_number:
                max_log_number = log_number
    return max_log_number + 1


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
