import random, math, argparse
import numpy as np
from numpy.random.mtrand import sample
from matplotlib import patches, pyplot as plt
import networkx as nx
from scipy import sparse
import constants as cn

set_dag_size = [i for i in range(6, 12)]  # random number of DAG  nodes
set_max_out = [1, 2, 3]  # max out_degree of one node
set_alpha = [0.5, 1.0, 1.5]  # DAG shape
set_beta = [0.0, 0.5, 1.0, 2.0]  # DAG regularity


class Node:

    def __init__(self, dag, idx):
        self.idx = idx
        self.data_size = None
        self.model_type = None
        self.in_edges = []
        self.out_edges = []
        self.is_finish = False
        self.allocated_xpu_fre = None
        self.ready_time = None  # 指依赖关系就绪，即前序任务都执行完且输入数据到达，而非资源就绪
        self.ready_time_esti = None
        self.resource_ready_time = None
        self.resource_ready_time_esti = None
        self.start_exe_time = None
        self.start_exe_time_esti = None
        self.finish_time = None
        self.finish_time_esti = None
        self.comp_delay = None
        self.trans_delay = None
        self.ES = None
        self.dag = dag
        self.energy = None


class Edge:
    def __init__(self, head, tail) -> None:
        self.head = head
        self.tail = tail
        self.weight = 0
        self.weight_tmp = None

    def to_nx(self):
        return (self.head.idx, self.tail.idx, {"weight": self.weight})


class DAG_task:

    def __init__(self, node_num):
        self.idx = None
        self.node_num = node_num - 2
        self.max_out = None
        self.alpha = None
        self.beta = None
        self.nodes = []
        self.edges = []
        # self.max_tole_delay = np.random.choice(cn.max_tole_delay_set)
        self.adj_matrix_np = None
        self.nx_obj = None
        self.create_time = None
        self.finish_time = None
        self.start_node_trans_delay = None
        self.UE = None
        self.energy = None

    def generate_pos(self, num, central_value, step):
        if num % 2 == 0:
            tmp = np.array(list(range(-num // 2, 0)) + list(range(1, num // 2 + 1))).astype(np.float32)
            tmp[tmp < 0] += 0.5
            tmp[tmp > 0] -= 0.5

            return central_value + step * tmp
        else:
            if num == 1:
                return np.array([central_value])
            else:
                return central_value + step * np.array(list(range(-(num - 1) // 2, num // 2 + 1)))

    def init_DAG(self, idx, max_out):
        self.idx = idx
        self.max_out = max_out
        self.alpha = random.choice(cn.alpha_set)
        self.beta = random.choice(cn.beta_set)

        length = math.floor(math.sqrt(self.node_num) / self.alpha)
        mean_value = self.node_num / length
        random_num = np.random.normal(loc=mean_value, scale=self.beta, size=(length, 1))

        # generate nodes in every layers
        generate_num = 0
        list_nodes_in_layer = []
        for i in range(len(random_num)):
            list_nodes_in_layer.append([])
            for j in range(math.ceil(random_num[i][0])):
                list_nodes_in_layer[i].append(j)
            generate_num += len(list_nodes_in_layer[i])

        if generate_num != self.node_num:
            if generate_num < self.node_num:
                for i in range(self.node_num - generate_num):
                    layer_ = random.randrange(0, length, 1)
                    list_nodes_in_layer[layer_].append(generate_num + i)
            if generate_num > self.node_num:
                i = generate_num - self.node_num
                while i > 0:
                    layer_ = random.randrange(0, length, 1)
                    if len(list_nodes_in_layer[layer_]) <= 1:
                        continue
                    else:
                        del list_nodes_in_layer[layer_][-1]
                        i -= 1

        # update dag node index and determain the position of subnodes
        updated_nodes_list = [[] for _ in range(length)]
        node_idx = 1
        position = {}
        pos_max_vertical = 20

        list_nodes_idx = ["start"] + [i for i in range(self.node_num)] + ["exit"]
        self.nodes = [Node(self, idx) for idx in list_nodes_idx]

        for i in range(length):
            idx_nodes_in_layer = list(range(node_idx, node_idx + len(list_nodes_in_layer[i])))
            for j in idx_nodes_in_layer:
                updated_nodes_list[i].append(self.nodes[j])
            node_idx += len(updated_nodes_list[i])
            pos_list = self.generate_pos(len(updated_nodes_list[i]), pos_max_vertical / 2, 4)
            for p, k in enumerate(updated_nodes_list[i]):
                position[k.idx] = (3 * (i + 1), pos_list[p])
        position[self.nodes[0].idx] = (0, pos_max_vertical / 2)  # start node
        position[self.nodes[-1].idx] = (3 * (length + 1), pos_max_vertical / 2)  # exit node

        # -----------------  connect nodes   ---------------------------

        for i in range(length - 1):
            nodes_in_next_layer = updated_nodes_list[i + 1]
            for node in updated_nodes_list[i]:
                tmp_od = random.randrange(1, self.max_out + 1, 1)
                tmp_od = min(len(updated_nodes_list[i + 1]), tmp_od)
                random_nodes_in_next_layer = random.sample(nodes_in_next_layer, tmp_od)  # 随机从下一层列表选择out_degree个节点进行连接
                for node_ in random_nodes_in_next_layer:
                    new_edge = Edge(node, node_)
                    self.edges.append(new_edge)
                    node.out_edges.append(new_edge)
                    node_.in_edges.append(new_edge)

        for node in self.nodes:  # 给所有没有入边的节点添加入口节点作父亲,node为列表中元素的位置，id为值
            if node.in_edges == [] and node.idx != "start" and node.idx != "exit":
                new_edge = Edge(self.nodes[0], node)
                self.edges.append(new_edge)
                self.nodes[0].out_edges.append(new_edge)
                node.in_edges.append(new_edge)

        for node in self.nodes:  # 给所有没有出边的节点添加出口节点作儿子
            if node.out_edges == [] and node.idx != "exit":
                new_edge = Edge(node, self.nodes[-1])
                self.edges.append(new_edge)
                node.out_edges.append(new_edge)
                self.nodes[-1].in_edges.append(new_edge)

        # -----------------初始化子节点属性，如数据大小，整体最大时延，节点偏好等

        for node in self.nodes:

            node.data_size = random.choice(cn.data_size_set)
            node.model_type = random.choice(cn.model_type)
            if node.idx != "start":
                for edge in node.in_edges:
                    edge.weight = node.data_size / len(node.in_edges)

        edge_list_nx = [edge.to_nx() for edge in self.edges]

        graph_nx = nx.DiGraph()
        graph_nx.add_edges_from(edge_list_nx)
        nx.draw_networkx(graph_nx, arrows=True, pos=position)
        adj_matrix = nx.adjacency_matrix(graph_nx)

        self.adj_matrix_np = adj_matrix.toarray()

        self.nx_obj = graph_nx

        plt.savefig("/home/hzx/Hete-DAG/figure/task_DAG" + str(idx) + ".png", format="PNG")
        plt.close()

        return self

    def search_for_successors(self, node, edges):
        """
        寻找后续节点
        :param node: 需要查找的节点id
        :param edges: DAG边信息(注意最好传列表的值(edges[:])进去而不是传列表的地址(edges)!!!)
        :return: node的后续节点id列表
        """
        map = {}
        if node == "exit":
            return print("error, 'Exit' node do not have successors!")
        for i in range(len(edges)):
            if edges[i][0] in map.keys():
                map[edges[i][0]].append(edges[i][1])
            else:
                map[edges[i][0]] = [edges[i][1]]
        pred = map[node]
        return pred

    def search_for_all_successors(self, node, edges):
        save = node
        node = [node]
        for ele in node:
            succ = self.search_for_successors(ele, edges)
            if len(succ) == 1 and succ[0] == "exit":
                break
            for item in succ:
                if item in node:
                    continue
                else:
                    node.append(item)
        node.remove(save)
        return node

    def search_for_predecessor(node, edges):
        """
        寻找前继节点
        :param node: 需要查找的节点id
        :param edges: DAG边信息
        :return: node的前继节点id列表
        """
        map = {}
        if node == "start":
            return print("error, 'Start' node do not have predecessor!")
        for i in range(len(edges)):
            if edges[i][1] in map.keys():
                map[edges[i][1]].append(edges[i][0])
            else:
                map[edges[i][1]] = [edges[i][0]]
        succ = map[node]
        return succ


# if __name__ == "__main__":
#     dag_task = DAG_tasks(
#         node_num_set=[6],
#         max_out_set=[2],
#         alpha_set=[0.8],
#         beta_set=[0.9],
#     )
#     dag = dag_task.init_DAG()
#     print("edges :", [(edge.head, edge.tail) for edge in dag["edges])
#     print("max_tole_delay :", dag["max_tole_delay)
#     print("data_size :", [node.data_size for node in dag["nodes])
#     print("adj_matrix :", dag["adj_matrix)
