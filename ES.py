from UE import UserEquipment
import constants as cn
import numpy as np
import copy
import logging
from colorama import init, Fore, Style, Back
import networkx as nx
import torch
from common import number_to_onehot
import bisect


class ES(object):

    def __init__(self, xpu_type, idx, slot_length, es_num):
        self.idx = idx
        self.slot_length = slot_length
        self.covered_ue = []
        self.xpu_type = xpu_type
        self.xpu_frequency = np.random.choice(cn.ES_xpu_frequency_set)
        self.tasks_wait_queue = []
        self.ready_tasks_queue = []
        self.computing_rest_time = []
        self.tasks_exe_queue = []
        self.next_task = None
        self.resource_status = [[0, 1]]
        self.one_hot = torch.tensor(number_to_onehot(idx, es_num, 1))

    # 更新资源状态
    def update_resource_status(self, task, mode):
        records = self.resource_status
        if mode == "allocate":
            times = [s[0] for s in records]
            insert_idx = bisect.bisect_left(times, task.start_exe_time)
            if insert_idx < len(times) and records[insert_idx][0] == task.start_exe_time:  # 已经存在时间戳
                records[insert_idx][1] = max(records[insert_idx - 1][1] - task.allocated_xpu_fre, 0)
            else:
                records.insert(
                    insert_idx,
                    [task.start_exe_time, max(records[insert_idx - 1][1] - task.allocated_xpu_fre, 0)],
                )
            for i in reversed(range(len(records))):
                if i > insert_idx:
                    records[i][1] = max(records[i][1] - task.allocated_xpu_fre, 0)
                else:
                    break

        elif mode == "release":
            times = [s[0] for s in records]
            insert_idx = bisect.bisect_left(times, task.finish_time)
            if insert_idx < len(times) and records[insert_idx][0] == task.finish_time:  # 已经存在时间戳
                records[insert_idx][1] = min(records[insert_idx - 1][1] + task.allocated_xpu_fre, 1)
            else:
                records.insert(
                    insert_idx,
                    [
                        task.finish_time,
                        min(records[insert_idx - 1][1] + task.allocated_xpu_fre, 1),
                    ],
                )
            for i in reversed(range(len(records))):
                if i > insert_idx:
                    records[i][1] = min(task.allocated_xpu_fre, 1)
                else:
                    break

    def step(self, time, slot_finished_task_queue, shortest_path_cache):
        # 记录资源状态

        tmp = self.resource_status[-1]
        self.resource_status = [[tmp[0], tmp[1]]]

        # 如果执行队列中有计算任务，进行step
        if self.tasks_wait_queue != [] or self.ready_tasks_queue != [] or self.tasks_exe_queue != []:

            # 检查在时隙结束（time + self.slot）之前，是否有任务ready,将符合要求的任务添加到ready队列
            if self.tasks_wait_queue != []:

                for task in self.tasks_wait_queue:  # 任务判断是否就绪（前序节点已经全部执行完成），若就绪，加入ready_queue

                    if task.idx == "start" and task.ready_time == None:
                        task.ready_time = task.dag.create_time + task.dag.start_node_trans_delay
                    elif task.idx != "start" and task.ready_time == None:
                        task.ready_time = max((edge.head.finish_time_esti + shortest_path_cache[edge.head.ES.idx][self.idx]) for edge in task.in_edges)

                while self.tasks_wait_queue:
                    task = self.tasks_wait_queue[0]  # 查看队首元素
                    if task.ready_time <= time + self.slot_length:
                        self.ready_tasks_queue.append(task)
                        self.tasks_wait_queue.remove(task)
                    else:
                        break  # 防止任务插队

            if self.tasks_exe_queue != []:  # 如果时隙内有任务执行完，释放资源,记录这些任务释放资源的时间点
                self.computing_rest_time = (np.array(self.computing_rest_time) - self.slot_length).tolist()

                temp = [v for v in self.computing_rest_time if v <= 0]
                # 当有任务计算完成时
                while temp != []:
                    i_less0 = self.computing_rest_time.index(temp[0])
                    finished_task = self.tasks_exe_queue[i_less0]
                    finished_task.finish_time = time + self.slot_length + self.computing_rest_time[i_less0]
                    if abs(finished_task.finish_time - finished_task.finish_time_esti) > 0.0001:
                        raise ValueError("估计的结束时间与真实不一致")
                    finished_task.is_finish = True
                    self.update_resource_status(finished_task, "release")

                    if finished_task.idx == "exit":
                        logging.warning(
                            Back.GREEN
                            + Fore.BLACK
                            + Style.BRIGHT
                            + "t= {},dag= {},node= {},complete at es-{}! total execution time= {}".format(
                                finished_task.finish_time,
                                finished_task.dag.create_time,
                                finished_task.idx,
                                self.idx,
                                finished_task.finish_time - finished_task.dag.create_time,
                            )
                            + Style.RESET_ALL
                        )
                    else:
                        logging.info(
                            Style.BRIGHT
                            + Fore.GREEN
                            + "t= {},dag= {},node= {},complete at es-{}! comp time= {}".format(
                                finished_task.finish_time,
                                finished_task.dag.create_time,
                                finished_task.idx,
                                self.idx,
                                finished_task.finish_time - finished_task.start_exe_time,
                            )
                            + Style.RESET_ALL
                        )
                    slot_finished_task_queue.put(finished_task)

                    del self.computing_rest_time[i_less0]
                    self.tasks_exe_queue.remove(finished_task)
                    del temp[0]

            # 将计算资源分配给就绪队列中的任务，并执行
            while self.ready_tasks_queue != []:
                # FCFS
                next_task = self.ready_tasks_queue[0]
                have_alloc_resource = False

                for status in reversed(self.resource_status):
                    # ========== 资源就绪时刻<任务的前序依赖就绪时刻，因此start_exe_time=ready_time(前序依赖就绪时间) ==========
                    if next_task.allocated_xpu_fre <= status[1]:
                        next_task.resource_ready_time = max(status[0], next_task.dag.create_time)
                        next_task.start_exe_time = max(next_task.ready_time, next_task.resource_ready_time)
                        self.tasks_exe_queue.append(next_task)
                        have_alloc_resource = True
                        self.ready_tasks_queue.remove(next_task)
                        self.update_resource_status(next_task, mode="allocate")
                        logging.info(
                            Style.BRIGHT
                            + Fore.MAGENTA
                            + "t= {}, dag= {},node= {}, start execute at es-{}, remaining time= {}".format(time, next_task.dag.create_time, next_task.idx, self.idx, next_task.comp_delay)
                            + Style.RESET_ALL
                        )

                        # 如果新执行的任务在时隙结束前执行完，释放资源
                        if next_task.start_exe_time + next_task.comp_delay <= time + self.slot_length:
                            next_task.finish_time = next_task.start_exe_time + next_task.comp_delay
                            next_task.is_finish = True
                            self.update_resource_status(next_task, mode="release")

                            if next_task.idx == "exit":
                                # reward.total_delay_panelty(next_task.finish_time - next_task.start_exe_time, next_task.dag.max_tole_delay)
                                logging.warning(
                                    Back.GREEN
                                    + Fore.BLACK
                                    + Style.BRIGHT
                                    + "t= {},dag= {},node= {}, complete at es-{}! total execution time= {}".format(
                                        next_task.finish_time,
                                        next_task.dag.create_time,
                                        next_task.idx,
                                        self.idx,
                                        next_task.finish_time - next_task.dag.create_time,
                                    )
                                    + Style.RESET_ALL
                                )
                            else:
                                logging.info(
                                    Style.BRIGHT
                                    + Fore.GREEN
                                    + "t= {},dag= {},node= {}, complete at es-{}, comp time= {}".format(
                                        next_task.finish_time,
                                        next_task.dag.create_time,
                                        next_task.idx,
                                        self.idx,
                                        next_task.finish_time - next_task.start_exe_time,
                                    )
                                    + Style.RESET_ALL
                                )

                            slot_finished_task_queue.put(next_task)
                            self.tasks_exe_queue.remove(next_task)
                        else:  # 新任务进入执行队列后在时隙结束前没有执行完，无需释放资源
                            self.computing_rest_time.append(next_task.start_exe_time + next_task.comp_delay - (time + self.slot_length))
                    else:
                        break  # 在遍历资源时只能从后往前找，因为FCFS下任务的最早检索的时刻即为资源可用时刻
                    if have_alloc_resource == False:
                        continue
                    else:
                        # 已经分配给任务计算资源，不需要重新检查后续status，导致重新分配
                        break
                if have_alloc_resource == False:  # 没有足够的资源可以分配给新任务，退出大循环
                    break
