# -*- coding: utf-8 -*-

import constants as cn
import numpy as np
from UE import UserEquipment
from ES import ES
import math
from common import number_to_onehot
from collections import deque
import logging
from colorama import init, Fore, Style, Back
from Edge_network import Edge_Network
import torch
import random
import networkx as nx

# 设置打印选项
np.set_printoptions(linewidth=500)
torch.set_printoptions(linewidth=500)


def generate_binary_code(max_value, value):
    # 计算需要的位数
    num_bits = max_value.bit_length()

    # 将数值转换为指定位数的二进制编码，使用 zfill 补齐
    binary_code = bin(value)[2:].zfill(num_bits)

    binary_tensor = torch.tensor([int(bit) for bit in binary_code])
    return binary_tensor


def postprocessing(action, edge_num, algorithm, greedy=False):

    if algorithm in ["QVPO", "MLP_DIPO"]:
        r_action = torch.zeros_like(action).squeeze(0)
        for i in range(len(action[0])):
            r_action[0] = math.floor(action[0][0] / (1 / edge_num)) if action[0][0] != 1.0 else edge_num - 1  # 卸载位置，处理边界情况，当value1为1.0时，向下取整映射
            assert 0 <= r_action[0] < edge_num, "动作转换超范围"
        r_action = r_action.unsqueeze(-1)
    else:
        r_action = torch.zeros_like(action)

        for i in range(len(action)):
            r_action[i][0] = math.floor(action[i][0] / (1 / edge_num)) if action[i][0] != 1.0 else edge_num - 1  # 卸载位置，处理边界情况，当value1为1.0时，向下取整映射
            assert 0 <= r_action[i][0] < edge_num, "动作转换超范围"

    return r_action


class Env:

    def __init__(self, args, dag_tasks_set):
        self.args = args
        self.slot_length = args.slot_length
        self.UEs_num = args.ue_num
        self.done = False
        self.reward = Reward(reward_type=args.reward_type, delay_coef=args.delay_coef, energy_coef=args.energy_coef, expected_delay_CVAR=args.expected_delay_CVAR)
        self.episode_len = args.max_ep_len
        self.expected_delay_CVAR = args.expected_delay_CVAR
        self.dag_tasks = dag_tasks_set
        self.edge_network = Edge_Network(args.slot_length, args.es_num)
        self.edge_network.init_edge_network()
        # 给每个ES添加UE
        i = 0
        for j in range(args.es_num):
            covered_ue_num = random.choice(cn.ue_num_set)
            for _ in range(covered_ue_num):
                tmp = UserEquipment(i)
                self.edge_network.ESs[j].covered_ue.append(tmp)
                tmp.ES = self.edge_network.ESs[j]
                i += 1

        self.UEs = [ue for ES in self.edge_network.ESs for ue in ES.covered_ue]  # create UEs

        self.max_cpu_num = 8
        self.max_xpu_num = 64

        self.total_job_num = 0
        self.finished_job_num = 0
        self.avg_total_delay = 0
        self.avg_energy = 0
        self.success_job_num = 0
        self.fail_job_num = 0
        self.success_rate = 0
        self.avg_trans_delay = 0
        self.avg_wait_delay = 0
        self.avg_comp_delay = 0
        self.episode_delay_record = []
        self.episode_energy_record = []
        # self.estimated_episode_delay_record = []
        # self.truely_episode_delay_record = [0 for _ in range(epsilon_len)]
        self.minus = None

    def reset(self):
        self.__init__(self.args, self.dag_tasks)

    def step(self, time, episode, new_task, action, algorithm, greedy=False):

        if action is not None:
            action = postprocessing(action, self.args.es_num, algorithm, greedy)
            logging.info(Style.BRIGHT + Fore.YELLOW + "episode={}, t= {}, make a decision {}".format(episode, time, action) + Style.RESET_ALL)

        slot_finished_tasks = self.edge_network.step(time, new_task, action, self.reward, self)

        for task in slot_finished_tasks:
            if task.idx == "exit":
                if task.finish_time - task.dag.create_time <= self.expected_delay_CVAR:
                    self.success_job_num += 1
                else:
                    self.fail_job_num += 1
                self.finished_job_num += 1
                # trans delay
                self.avg_trans_delay = (
                    self.avg_trans_delay * (self.finished_job_num - 1) + sum([nx.shortest_path_length(self.edge_network.nx_obj, source=edge.head.ES.idx, target=edge.tail.ES.idx, weight="weight") for edge in task.dag.edges])
                ) / self.finished_job_num
                # computing delay
                self.avg_comp_delay = (self.avg_comp_delay * (self.finished_job_num - 1) + sum([node.comp_delay for node in task.dag.nodes])) / self.finished_job_num
                # total delay
                self.avg_total_delay = (self.avg_total_delay * (self.finished_job_num - 1) + (task.finish_time - task.dag.create_time)) / self.finished_job_num
                # self.truely_episode_delay_record[round(task.dag.create_time / self.slot)] = task.finish_time - task.dag.create_time
                # self.minus = [self.truely_episode_delay_record[i] - self.estimated_episode_delay_record[i] for i in range(len(self.estimated_episode_delay_record))]
                # queue delay
                self.avg_wait_delay = self.avg_total_delay - self.avg_comp_delay - self.avg_trans_delay
                # energy
                dag_energy = sum(subtask.energy for subtask in task.dag.nodes)
                self.avg_energy = (self.avg_energy * (self.finished_job_num - 1) + dag_energy) / self.finished_job_num

                self.episode_delay_record.append(task.finish_time - task.dag.create_time)
                self.episode_energy_record.append(dag_energy)

        self.success_rate = self.success_job_num / self.finished_job_num if self.finished_job_num != 0 else 0

    def get_dag_task_status(self, dag, device):

        # 找到非零元素的掩码
        non_zero_mask = dag.adj_matrix_np != 0

        # 提取非零元素
        non_zero_elements = dag.adj_matrix_np[non_zero_mask]

        # 计算非零元素的最小值和最大值
        min_val = np.min(non_zero_elements)
        max_val = np.max(non_zero_elements)

        # 对非零元素进行归一化
        normalized_non_zero_elements = 0.005 + ((non_zero_elements - min_val) / (max_val - min_val + 1e-7)) * (0.01 - 0.005)

        # 创建归一化后的邻接矩阵并将非零元素替换
        normalized_adj_matrix = np.zeros_like(dag.adj_matrix_np, dtype=float)
        normalized_adj_matrix[non_zero_mask] = normalized_non_zero_elements

        dag_adj_matrix = torch.tensor(normalized_adj_matrix, dtype=torch.float32).to(device)
        dag_edge_index = dag_adj_matrix.nonzero(as_tuple=False).t()
        edge_weights = dag_adj_matrix[dag_edge_index[0], dag_edge_index[1]]

        max_data_size = max(cn.data_size_set)
        data_size = np.array([node.data_size / max_data_size for node in dag.nodes]).astype(np.float32)
        data_size = np.expand_dims(data_size, axis=-1)

        model_type = np.array([number_to_onehot(node.model_type, max(cn.model_type), min(cn.model_type)) for node in dag.nodes]).astype(np.float32)

        nearest_es = dag.UE.ES.one_hot
        nearest_es_repeated = nearest_es.unsqueeze(0).repeat(len(dag.nodes), 1)  # [N,]->[1,N]->[node_num,N]

        # max_tole_delay = np.array([dag.max_tole_delay / max(cn.max_tole_delay_set)]).astype(np.float32)
        # max_tole_delay = np.expand_dims(max_tole_delay, axis=-1)
        # max_tole_delay_repeated = max_tole_delay.repeat(len(dag.nodes), 0)  # [1,]->[1,1]->[node_num,1]

        feature = torch.tensor(np.hstack((data_size, model_type, nearest_es_repeated)), dtype=torch.float32).to(device)  # [num_node, num_feature(1+4+N)]

        logging.debug("dag's edge index: {}".format(dag_edge_index))
        logging.debug("dag's edge weights: {}".format(edge_weights))
        logging.debug("dag's feature matrix: {}".format(feature))

        return dag_edge_index, edge_weights, feature

    def get_dag_task_status_no_type(self, dag, device):

        # 找到非零元素的掩码
        non_zero_mask = dag.adj_matrix_np != 0

        # 提取非零元素
        non_zero_elements = dag.adj_matrix_np[non_zero_mask]

        # 计算非零元素的最小值和最大值
        min_val = np.min(non_zero_elements)
        max_val = np.max(non_zero_elements)

        # 对非零元素进行归一化
        normalized_non_zero_elements = 0.005 + ((non_zero_elements - min_val) / (max_val - min_val + 1e-7)) * (0.01 - 0.005)

        # 创建归一化后的邻接矩阵并将非零元素替换
        normalized_adj_matrix = np.zeros_like(dag.adj_matrix_np, dtype=float)
        normalized_adj_matrix[non_zero_mask] = normalized_non_zero_elements

        dag_adj_matrix = torch.tensor(normalized_adj_matrix, dtype=torch.float32).to(device)
        dag_edge_index = dag_adj_matrix.nonzero(as_tuple=False).t()
        edge_weights = dag_adj_matrix[dag_edge_index[0], dag_edge_index[1]]

        max_data_size = max(cn.data_size_set)
        data_size = np.array([node.data_size / max_data_size for node in dag.nodes]).astype(np.float32)
        data_size = np.expand_dims(data_size, axis=-1)

        nearest_es = dag.UE.ES.one_hot
        nearest_es_repeated = nearest_es.unsqueeze(0).repeat(len(dag.nodes), 1)  # [N,]->[1,N]->[node_num,N]

        # max_tole_delay = np.array([dag.max_tole_delay / max(cn.max_tole_delay_set)]).astype(np.float32)
        # max_tole_delay = np.expand_dims(max_tole_delay, axis=-1)
        # max_tole_delay_repeated = max_tole_delay.repeat(len(dag.nodes), 0)  # [1,]->[1,1]->[node_num,1]

        feature = torch.tensor(np.hstack((data_size, nearest_es_repeated)), dtype=torch.float32).to(device)  # [num_node, num_feature(1+N)]

        logging.debug("dag's edge index: {}".format(dag_edge_index))
        logging.debug("dag's edge weights: {}".format(edge_weights))
        logging.debug("dag's feature matrix: {}".format(feature))

        return dag_edge_index, edge_weights, feature

    def get_ue_wait_queue_status(self):
        queue_len = []
        for i in range(self.UEs_num):
            queue_len.append(len(self.UEs[i].wait_trans_queue))

        logging.debug("ue wait queue length: {}".format(np.array(queue_len)))
        return np.array(queue_len) / 10

    def get_es_anly_queue_status(self):
        logging.debug("es anly queue length: {}".format(np.array(len(self.ES.tasks_exe_queue))))
        return np.array([len(self.ES.tasks_exe_queue)])


def calculate_VaR_CVaR(data, alpha=0.95):
    """
    计算给定数据集和置信水平(alpha)的CVaR（条件风险价值）。

    :param (list/array) data:  损失值列表（正值表示损失，负值表示收益需预处理）。
    :param (float) alpha:  置信水平（0 < alpha < 1，例如0.95表示95%置信水平）。

    :return  CVaR值 (float):
    """
    var = np.quantile(data, alpha)
    tail = data[data >= var]
    return var, np.mean(tail) if len(tail) > 0 else var


class Reward(object):

    def __init__(self, reward_type, delay_coef, energy_coef, expected_delay_CVAR):
        self.reward_type = reward_type
        self.delay_coef = delay_coef
        self.energy_coef = energy_coef
        self.expected_delay_CVAR = expected_delay_CVAR
        self.episode_delay_record = []
        self.last_SortinoRatio = 0
        self.current_SortinoRatio = 0
        self.last_SharpeRatio = 0
        self.current_SharpeRatio = 0
        self.down_variance = 0
        self.variance = 0
        self.current_average_delay = 0

        self.delay_reward = 0
        self.queue_penalty = 0
        self.load_penalty = 0
        self.energy_reward = 0
        self.idea_queue_len = 2

    def delay_panelty(self, delay):
        self.episode_delay_record.append(delay)
        np_dealy_record = np.array(self.episode_delay_record)

        self.current_average_delay = np.mean(np_dealy_record)
        _, current_cvar_value = calculate_VaR_CVaR(np_dealy_record, alpha=0.95)

        if self.reward_type == 0:
            # 对比 Sortino Rate(t)和Sortino Rate(t-1)，给予固定惩罚
            self.down_variance = np.sqrt(np.mean(np.maximum(current_cvar_value - self.expected_delay_CVAR, 0) ** 2) + 1e-7)
            self.current_SortinoRatio = -self.current_average_delay / self.down_variance

            self.delay_reward = 1 if self.current_SortinoRatio >= self.last_SortinoRatio else 0
            self.last_SortinoRatio = self.current_SortinoRatio

        elif self.reward_type == 1:  #
            # 优化Sharpe率 = 时延均值/时延样本之间的l2范数（方差/sqar（n）） ==> 对比 SR(t)和SR(t-1)，给予固定惩罚
            # 论文（Towards Risk-Averse Edge Computing With Deep Reinforcement Learning）的方差计算方式
            self.variance = np.linalg.norm(np_dealy_record - self.current_average_delay) + 1e-7
            self.current_SharpeRatio = -self.current_average_delay / self.variance

            self.delay_reward = 1 if self.current_SharpeRatio >= self.last_SharpeRatio else 0
            self.last_SharpeRatio = self.current_SharpeRatio

        elif self.reward_type == 2:
            # opt = -时延均值
            self.delay_reward = -self.current_average_delay

    def queue_panelty(self, queue_length):
        self.queue_penalty -= 0.01 * queue_length

    def load_banlence_penalty(self, list_of_load):
        # 计算平均值
        mean = sum(list_of_load) / len(list_of_load)
        # 计算方差
        variance = sum((x - mean) ** 2 for x in list_of_load) / len(list_of_load)
        self.load_penalty -= 0.0000005 * variance

    def energy_penalty(self, value):
        self.energy_reward -= value

    def reset(self):
        self.delay_reward = 0
        self.queue_penalty = 0
        self.load_penalty = 0
        self.energy_reward = 0
        self.idea_queue_len = 2

    def get_value(self):
        return self.delay_coef * self.delay_reward + self.energy_coef * self.energy_reward

    def get_reward_value(self):
        return self.energy_reward

    def get_cost_value(self):
        return self.delay_reward


# if __name__ == "__main__":

#     a = Env_system(1, 1, 200)
#     b, c, d = a.get_dag_task_status(a.dag_tasks[0])
#     print("edges :", [(edge.head, edge.tail) for edge in a.dag_tasks[0]["edges"]])
#     print(b)
#     print(c)
#     print(d)
#     print(d.shape)
