import networkx as nx
import numpy as np

import constants as cn
from ES import ES
import torch
import logging
import math

import concurrent.futures
from queue import Queue
import bisect
from common import number_to_onehot


class vir_ES:
    def __init__(self, idx):
        self.idx = idx
        self.resource_record = [[0, 1]]  # (time，rest_resource)


class Edge_Network:

    def __init__(self, slot_length, server_num):
        self.slot_length = slot_length
        self.es_num = server_num
        self.ue_num_list = []
        self.ESs = []
        self.vir_ESs = [vir_ES(i) for i in range(server_num)]
        self.adj_matrix = None
        self.edges = None
        self.comp_resource = []
        self.comp_free = []
        self.info = None
        self.nx_obj = None
        self.slot_finish_tasks = []
        self.shortest_path_cache = None

    def init_edge_network(self):
        """
        output => G(networkx objective), G's adjacency matrix
        """
        # 创建一个空的无向图
        G = nx.Graph()
        if self.es_num == 10:
            # idx_ = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            idx_ = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            G.add_nodes_from(idx_)

            # 批量添加边
            self.edges = [
                (1, 2, {"weight": 4 * cn.ms, "color": "red", "style": "-"}),
                (1, 3, {"weight": 4 * cn.ms, "color": "red", "style": "-"}),
                (1, 4, {"weight": 4 * cn.ms, "color": "red", "style": "-"}),
                (2, 10, {"weight": 1.41 * cn.ms, "color": "green", "style": "--"}),
                (2, 9, {"weight": 1.41 * cn.ms, "color": "green", "style": "--"}),
                (9, 10, {"weight": 2 * cn.ms, "color": "black", "style": ":"}),
                (3, 8, {"weight": 1.41 * cn.ms, "color": "green", "style": "--"}),
                (3, 7, {"weight": 1.41 * cn.ms, "color": "green", "style": "--"}),
                (7, 8, {"weight": 2 * cn.ms, "color": "black", "style": ":"}),
                (4, 5, {"weight": 1.41 * cn.ms, "color": "green", "style": "--"}),
                (4, 6, {"weight": 1.41 * cn.ms, "color": "green", "style": "--"}),
                (5, 6, {"weight": 2 * cn.ms, "color": "black", "style": ":"}),
            ]
        elif self.es_num == 15:
            idx_ = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            G.add_nodes_from(idx_)

            # 批量添加边
            self.edges = [
                (1, 2, {"weight": 4 * cn.ms, "color": "red", "style": "-"}),
                (1, 3, {"weight": 4 * cn.ms, "color": "red", "style": "-"}),
                (2, 3, {"weight": 4 * cn.ms, "color": "red", "style": "-"}),
                (1, 4, {"weight": 1.41 * cn.ms, "color": "green", "style": "--"}),
                (1, 5, {"weight": 1.41 * cn.ms, "color": "green", "style": "--"}),
                (1, 6, {"weight": 1.41 * cn.ms, "color": "green", "style": "--"}),
                (1, 7, {"weight": 1.41 * cn.ms, "color": "green", "style": "--"}),
                (4, 5, {"weight": 2 * cn.ms, "color": "black", "style": ":"}),
                (4, 6, {"weight": 2 * cn.ms, "color": "black", "style": ":"}),
                (5, 7, {"weight": 2 * cn.ms, "color": "black", "style": ":"}),
                (6, 7, {"weight": 2 * cn.ms, "color": "black", "style": ":"}),
                (2, 8, {"weight": 1.41 * cn.ms, "color": "green", "style": "--"}),
                (2, 9, {"weight": 1.41 * cn.ms, "color": "green", "style": "--"}),
                (2, 10, {"weight": 1.41 * cn.ms, "color": "green", "style": "--"}),
                (2, 11, {"weight": 1.41 * cn.ms, "color": "green", "style": "--"}),
                (8, 9, {"weight": 2 * cn.ms, "color": "black", "style": ":"}),
                (8, 10, {"weight": 2 * cn.ms, "color": "black", "style": ":"}),
                (9, 11, {"weight": 2 * cn.ms, "color": "black", "style": ":"}),
                (10, 11, {"weight": 2 * cn.ms, "color": "black", "style": ":"}),
                (3, 12, {"weight": 1.41 * cn.ms, "color": "green", "style": "--"}),
                (3, 13, {"weight": 1.41 * cn.ms, "color": "green", "style": "--"}),
                (3, 14, {"weight": 1.41 * cn.ms, "color": "green", "style": "--"}),
                (3, 15, {"weight": 1.41 * cn.ms, "color": "green", "style": "--"}),
                (12, 13, {"weight": 2 * cn.ms, "color": "black", "style": ":"}),
                (12, 14, {"weight": 2 * cn.ms, "color": "black", "style": ":"}),
                (13, 15, {"weight": 2 * cn.ms, "color": "black", "style": ":"}),
                (14, 15, {"weight": 2 * cn.ms, "color": "black", "style": ":"}),
            ]
        elif self.es_num == 20:
            idx_ = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            G.add_nodes_from(idx_)

            # 批量添加边
            self.edges = [
                (1, 2, {"weight": 4 * cn.ms, "color": "red", "style": "-"}),
                (1, 3, {"weight": 4 * cn.ms, "color": "red", "style": "-"}),
                (2, 4, {"weight": 4 * cn.ms, "color": "red", "style": "-"}),
                (3, 4, {"weight": 4 * cn.ms, "color": "red", "style": "-"}),
                (1, 5, {"weight": 1.41 * cn.ms, "color": "green", "style": "--"}),
                (1, 6, {"weight": 1.41 * cn.ms, "color": "green", "style": "--"}),
                (1, 7, {"weight": 1.41 * cn.ms, "color": "green", "style": "--"}),
                (1, 8, {"weight": 1.41 * cn.ms, "color": "green", "style": "--"}),
                (5, 6, {"weight": 2 * cn.ms, "color": "black", "style": ":"}),
                (5, 7, {"weight": 2 * cn.ms, "color": "black", "style": ":"}),
                (6, 8, {"weight": 2 * cn.ms, "color": "black", "style": ":"}),
                (7, 8, {"weight": 2 * cn.ms, "color": "black", "style": ":"}),
                (2, 9, {"weight": 1.41 * cn.ms, "color": "green", "style": "--"}),
                (2, 10, {"weight": 1.41 * cn.ms, "color": "green", "style": "--"}),
                (2, 11, {"weight": 1.41 * cn.ms, "color": "green", "style": "--"}),
                (2, 12, {"weight": 1.41 * cn.ms, "color": "green", "style": "--"}),
                (9, 10, {"weight": 2 * cn.ms, "color": "black", "style": ":"}),
                (9, 11, {"weight": 2 * cn.ms, "color": "black", "style": ":"}),
                (10, 12, {"weight": 2 * cn.ms, "color": "black", "style": ":"}),
                (11, 12, {"weight": 2 * cn.ms, "color": "black", "style": ":"}),
                (3, 13, {"weight": 1.41 * cn.ms, "color": "green", "style": "--"}),
                (3, 14, {"weight": 1.41 * cn.ms, "color": "green", "style": "--"}),
                (3, 15, {"weight": 1.41 * cn.ms, "color": "green", "style": "--"}),
                (3, 16, {"weight": 1.41 * cn.ms, "color": "green", "style": "--"}),
                (13, 14, {"weight": 2 * cn.ms, "color": "black", "style": ":"}),
                (13, 15, {"weight": 2 * cn.ms, "color": "black", "style": ":"}),
                (14, 16, {"weight": 2 * cn.ms, "color": "black", "style": ":"}),
                (15, 16, {"weight": 2 * cn.ms, "color": "black", "style": ":"}),
                (4, 17, {"weight": 1.41 * cn.ms, "color": "green", "style": "--"}),
                (4, 18, {"weight": 1.41 * cn.ms, "color": "green", "style": "--"}),
                (4, 19, {"weight": 1.41 * cn.ms, "color": "green", "style": "--"}),
                (4, 20, {"weight": 1.41 * cn.ms, "color": "green", "style": "--"}),
                (17, 18, {"weight": 2 * cn.ms, "color": "black", "style": ":"}),
                (17, 19, {"weight": 2 * cn.ms, "color": "black", "style": ":"}),
                (18, 20, {"weight": 2 * cn.ms, "color": "black", "style": ":"}),
                (19, 20, {"weight": 2 * cn.ms, "color": "black", "style": ":"}),
            ]
        else:
            raise NotImplementedError

        G.add_edges_from(self.edges)

        self.adj_matrix = nx.to_numpy_array(G, weight="weight")

        self.nx_obj = G
        # 缓存所有最短路径
        self.shortest_path_cache = dict(nx.all_pairs_dijkstra_path_length(self.nx_obj))

        # 为边缘节点初始化计算资源
        self.ESs = [ES(cn.es_resource_type[i], i + 1, self.slot_length, self.es_num) for i in range(self.es_num)]  # index从1开始，与创建nx图时的序号一致

        assert self.es_num == len(self.adj_matrix) == len(self.ESs), "维度不匹配"

        self.update_info()

    def update_info(self):
        self.info = {
            "adj_matrix": self.adj_matrix,
            "xpu_type": np.array([number_to_onehot(self.ESs[i].xpu_type, cn.resource_type - 1, 0) for i in range(self.es_num)], dtype=np.float32),
            "tasks_wait_queue_len": [len(self.ESs[i].tasks_wait_queue) for i in range(self.es_num)],
            "tasks_ready_queue_len": [len(self.ESs[i].ready_tasks_queue) for i in range(self.es_num)],
            "tasks_exe_queue_len": [len(self.ESs[i].tasks_exe_queue) for i in range(self.es_num)],
            "xpu_free": np.array([self.ESs[i].resource_status[-1][1] for i in range(self.es_num)], dtype=np.float32),
        }

    def get_state(self, device):

        self.update_info()

        edge_net_adj_matrix = torch.tensor(self.info["adj_matrix"], dtype=torch.float32).to(device)
        network_edge_index = edge_net_adj_matrix.nonzero(as_tuple=False).t()
        edge_weights = edge_net_adj_matrix[network_edge_index[0], network_edge_index[1]]

        edge_net_xpu_type = np.array(self.info["xpu_type"]).astype(np.float32)

        edge_net_xpu_free = self.info["xpu_free"]
        edge_net_xpu_free = np.expand_dims(edge_net_xpu_free, axis=-1)

        # edge_net_covered_ue_num = self.info["covered_ue_num"]
        # edge_net_covered_ue_num = np.expand_dims(edge_net_covered_ue_num, axis=-1)

        edge_net_task_wait_queue = np.array([math.log2(self.info["tasks_wait_queue_len"][i] + 1) / 10 for i in range(self.es_num)]).astype(np.float32)
        edge_net_task_wait_queue = np.expand_dims(edge_net_task_wait_queue, axis=-1)

        edge_net_task_ready_queue = np.array([math.log2(self.info["tasks_ready_queue_len"][i] + 1) / 10 for i in range(self.es_num)]).astype(np.float32)
        edge_net_task_ready_queue = np.expand_dims(edge_net_task_ready_queue, axis=-1)

        edge_net_task_exe_queue = np.array([math.log2(self.info["tasks_exe_queue_len"][i] + 1) / 10 for i in range(self.es_num)]).astype(np.float32)
        edge_net_task_exe_queue = np.expand_dims(edge_net_task_exe_queue, axis=-1)

        feature = torch.tensor(np.hstack((edge_net_xpu_type, edge_net_xpu_free, edge_net_task_wait_queue, edge_net_task_ready_queue, edge_net_task_exe_queue)), dtype=torch.float32).to(device)  # [num_node, num_feature(3+1+1+1+1)]

        logging.debug("edge network's edge index: {}".format(network_edge_index))
        logging.debug("edge network's edge weights: {}".format(edge_weights))
        logging.debug("edge network's feature matrix: {}".format(feature))

        return network_edge_index, edge_weights, feature

    def get_state_no_type(self, device):

        self.update_info()

        edge_net_adj_matrix = torch.tensor(self.info["adj_matrix"], dtype=torch.float32).to(device)
        network_edge_index = edge_net_adj_matrix.nonzero(as_tuple=False).t()
        edge_weights = edge_net_adj_matrix[network_edge_index[0], network_edge_index[1]]

        edge_net_xpu_free = self.info["xpu_free"]
        edge_net_xpu_free = np.expand_dims(edge_net_xpu_free, axis=-1)

        # edge_net_covered_ue_num = self.info["covered_ue_num"]
        # edge_net_covered_ue_num = np.expand_dims(edge_net_covered_ue_num, axis=-1)

        edge_net_task_wait_queue = np.array([math.log2(self.info["tasks_wait_queue_len"][i] + 1) / 10 for i in range(self.es_num)]).astype(np.float32)
        edge_net_task_wait_queue = np.expand_dims(edge_net_task_wait_queue, axis=-1)

        edge_net_task_ready_queue = np.array([math.log2(self.info["tasks_ready_queue_len"][i] + 1) / 10 for i in range(self.es_num)]).astype(np.float32)
        edge_net_task_ready_queue = np.expand_dims(edge_net_task_ready_queue, axis=-1)

        edge_net_task_exe_queue = np.array([math.log2(self.info["tasks_exe_queue_len"][i] + 1) / 10 for i in range(self.es_num)]).astype(np.float32)
        edge_net_task_exe_queue = np.expand_dims(edge_net_task_exe_queue, axis=-1)

        feature = torch.tensor(np.hstack((edge_net_xpu_free, edge_net_task_wait_queue, edge_net_task_ready_queue, edge_net_task_exe_queue)), dtype=torch.float32).to(device)  # [num_node, num_feature(1+1+1+1)]

        logging.debug("edge network's edge index: {}".format(network_edge_index))
        logging.debug("edge network's edge weights: {}".format(edge_weights))
        logging.debug("edge network's feature matrix: {}".format(feature))

        return network_edge_index, edge_weights, feature

    def step(self, time, new_task, actions, reward, env):

        if new_task is not None and actions is not None:
            assert len(new_task.nodes) == actions.shape[0], "action长度与DAG子任务数不匹配"

            # 新增任务
            for i in range(actions.shape[0]):

                new_task.nodes[i].ES = self.ESs[int(actions[i][0])]
                new_task.nodes[i].allocated_xpu_fre = 1
                # computing delay
                new_task.nodes[i].comp_delay = cn.model_delay_matrix[new_task.nodes[i].model_type][new_task.nodes[i].ES.xpu_type]
                # energy
                new_task.nodes[i].energy = cn.model_energy_matrix[new_task.nodes[i].model_type][new_task.nodes[i].ES.xpu_type]

            # start 节点的传输时延
            new_task.start_node_trans_delay = new_task.nodes[0].data_size / cn.band_width + self.shortest_path_cache[new_task.UE.ES.idx][new_task.nodes[0].ES.idx]

            # 在决策之后直接进行时延计算，给出即时奖励
            for edge in new_task.edges:
                edge.weight_tmp = self.shortest_path_cache[edge.head.ES.idx][edge.tail.ES.idx]

            for i in range(actions.shape[0]):
                self.ESs[int(actions[i][0])].tasks_wait_queue.append(new_task.nodes[i])
                resource_ready_time, wait_delay = self.get_ES_wait_time(time, new_task.nodes[i].ES.idx - 1, new_task.nodes[i])  # ES的index从1开始，所以减1
                if new_task.nodes[i].idx == "start":
                    new_task.nodes[i].finish_time_esti = new_task.nodes[i].start_exe_time_esti + new_task.nodes[i].comp_delay
                else:
                    new_task.nodes[i].finish_time_esti = new_task.nodes[i].start_exe_time_esti + new_task.nodes[i].comp_delay

            total_energy = self.get_dag_energy(new_task)
            new_task.energy = total_energy
            reward.energy_penalty(total_energy)

            total_delay = max([node.finish_time_esti for node in new_task.nodes]) - new_task.create_time

            reward.delay_panelty(total_delay)
            # env.estimated_episode_delay_record.append(total_delay)
        self.slot_finish_tasks = []

        list_of_queue_len = [len(es.tasks_wait_queue) + len(es.ready_tasks_queue) for es in self.ESs]
        reward.load_banlence_penalty(list_of_queue_len)

        # reward.queue_panelty(len(ES.tasks_wait_queue) + len(ES.ready_tasks_queue))

        # ES step （多线程并行step，然后收集结果）
        # 创建线程安全的任务收集队列
        step_queue = Queue()

        # 使用上下文管理器确保线程池资源释放
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.es_num) as executor:
            # 创建所有ES的step任务
            futures = {executor.submit(ES.step, time, step_queue, self.shortest_path_cache): ES for ES in self.ESs}  # 使用独立队列

            # 等待本时间步所有ES完成
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()  # 显式获取结果确保异常抛出
                except Exception as e:
                    executor.shutdown(wait=False, cancel_futures=True)  # 立即终止所有线程
                    raise

        # 合并本步结果到主队列
        while not step_queue.empty():
            self.slot_finish_tasks.append(step_queue.get())

        return self.slot_finish_tasks

    def get_ideal_dag_delay(self, new_task):
        # 不考虑边缘节点资源约束下的排队时延，等待时延仅可能来源于前序节点未完成。
        for edge in new_task.edges:
            edge.weight_tmp = self.shortest_path_cache[edge.head.ES.idx][edge.tail.ES.idx]

        for node in new_task.nodes:
            if node.idx == "start":
                node.finish_time_esti = node.comp_delay
            else:
                node.finish_time_esti = max([(edge.head.finish_time_esti + edge.weight_tmp) for edge in node.in_edges]) + node.comp_delay

        total_delay = max([node.finish_time_esti for node in new_task.nodes])

        return total_delay

    def get_ES_wait_time(self, time, es_idx, task):

        comp_delay, allocated_resource = task.comp_delay, task.allocated_xpu_fre
        records = self.vir_ESs[es_idx].resource_record
        # 找到满足资源要求的最小时间点
        target_record = None
        for k in reversed(range(len(records))):
            if records[k][1] >= allocated_resource:
                target_record = records[k]
            else:
                break

        resource_ready_time = max(target_record[0], time)
        wait_delay = resource_ready_time - time
        task.resource_ready_time_esti = resource_ready_time
        if task.idx == "start":
            task.ready_time_esti = task.dag.create_time + task.dag.start_node_trans_delay

            task.start_exe_time_esti = max(task.ready_time_esti, resource_ready_time)
        else:
            preorder_node_finish_time = max([(edge.head.finish_time_esti + edge.weight_tmp) for edge in task.in_edges])
            task.ready_time_esti = preorder_node_finish_time
            task.start_exe_time_esti = max(task.ready_time_esti, resource_ready_time)
        # 更新资源记录
        task_finish_time = task.start_exe_time_esti + comp_delay
        if target_record[0] < task.start_exe_time_esti:
            times = [s[0] for s in records]
            insert_idx = bisect.bisect_left(times, task.start_exe_time_esti)
            records.insert(
                insert_idx,
                [task.start_exe_time_esti, records[insert_idx - 1][1] - allocated_resource],
            )
            for i in reversed(range(len(records))):
                if i > insert_idx:
                    records[i][1] = max(records[i][1] - allocated_resource, 0)
                else:
                    break

            # 增加执行结束的记录
            times = [s[0] for s in records]
            insert_idx = bisect.bisect_left(times, task_finish_time)
            records.insert(
                insert_idx,
                [
                    task_finish_time,
                    min(records[insert_idx - 1][1] + allocated_resource, 1),
                ],
            )
            for i in reversed(range(len(records))):
                if i > insert_idx:
                    records[i][1] += allocated_resource
                else:
                    break
        elif target_record[0] >= time:
            times = [s[0] for s in records]
            insert_idx = bisect.bisect_left(times, target_record[0])
            for i in reversed(range(len(records))):
                if i >= insert_idx:
                    records[i][1] = max(records[i][1] - allocated_resource, 0)
                else:
                    break

            # 增加执行结束的记录
            times = [s[0] for s in records]
            insert_idx = bisect.bisect_left(times, task_finish_time)
            records.insert(
                insert_idx,
                [
                    task_finish_time,
                    min(records[insert_idx - 1][1] + allocated_resource, 1),
                ],
            )
            for i in reversed(range(len(records))):
                if i > insert_idx:
                    records[i][1] = min(records[i][1] + allocated_resource, 1)
                else:
                    break

        return resource_ready_time, wait_delay

    def get_queue_position(self, new_task):
        total_queue_len = 0
        for node in new_task.nodes:
            total_queue_len += len(node.ES.tasks_exe_queue) + len(node.ES.ready_tasks_queue) + len(node.ES.tasks_wait_queue) - 1
        for edge in new_task.edges:
            if edge.head.ES.idx == edge.tail.ES.idx:
                total_queue_len -= 1
        return total_queue_len

    def get_dag_energy(self, new_task):
        total_energy = 0
        for node in new_task.nodes:
            total_energy += node.energy
        return total_energy


# if __name__ == "__main__":
#     a = Edge_Network()
#     info = a.init_edge_network()
#     print(info)
