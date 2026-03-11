import argparse
import ast
import csv
import random
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

import constants as cn


class DatasetNode:

    def __init__(self, dag, idx):
        self.idx = idx
        self.data_size = None
        self.model_type = None
        self.in_edges = []
        self.out_edges = []
        self.is_finish = False
        self.allocated_xpu_fre = None
        self.ready_time = None
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
        self.info = None


class DatasetEdge:

    def __init__(self, head, tail) -> None:
        self.head = head
        self.tail = tail
        self.weight = 0
        self.weight_tmp = None

    def to_nx(self):
        return (self.head.idx, self.tail.idx, {"weight": self.weight})


class DatasetDAGTask:

    def __init__(self, node_num: int):
        self.idx = None
        self.node_num = node_num - 2
        self.max_out = None
        self.info = {}
        self.nodes = []
        self.edges = []
        self.adj_matrix_np = None
        self.nx_obj = None
        self.create_time = None
        self.finish_time = None
        self.start_node_trans_delay = None
        self.UE = None
        self.energy = None


class DatasetDAGTaskBuilder:
    """根据离线数据集构造可直接用于环境的 DAG 任务集合。"""

    CLUSTER_FIELDNAMES = [
        "task_name",
        "instance_num",
        "job_name",
        "task_type",
        "status",
        "start_time",
        "end_time",
        "plan_cpu",
        "plan_mem",
    ]

    CLUSTER_DAG_PATTERN = re.compile(r"^[A-Za-z]+(\d+(?:_\d+)*)$")

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.random = random.Random(seed)

    def build(
        self,
        dataset_name: str,
        file_path: str,
        dag_num: Optional[int] = None,
        target_total_nodes: Optional[int] = None,
        min_total_nodes: Optional[int] = None,
        max_total_nodes: Optional[int] = None,
        shuffle: bool = True,
    ) -> List[DatasetDAGTask]:
        dataset_name = dataset_name.lower()
        if dataset_name in {"huawei", "huawei-network-ai-challenge"}:
            return self.build_from_huawei(
                file_path=file_path,
                dag_num=dag_num,
                target_total_nodes=target_total_nodes,
                min_total_nodes=min_total_nodes,
                max_total_nodes=max_total_nodes,
                shuffle=shuffle,
            )
        if dataset_name in {"cluster", "alibaba", "cluster-trace-v2018"}:
            return self.build_from_cluster(
                file_path=file_path,
                dag_num=dag_num,
                target_total_nodes=target_total_nodes,
                min_total_nodes=min_total_nodes,
                max_total_nodes=max_total_nodes,
            )
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    def build_task(
        self,
        dataset_name: str,
        file_path: str,
        index: int = 0,
        target_total_nodes: Optional[int] = None,
        min_total_nodes: Optional[int] = None,
        max_total_nodes: Optional[int] = None,
        shuffle: bool = True,
    ) -> DatasetDAGTask:
        dag_tasks = self.build(
            dataset_name=dataset_name,
            file_path=file_path,
            dag_num=index + 1,
            target_total_nodes=target_total_nodes,
            min_total_nodes=min_total_nodes,
            max_total_nodes=max_total_nodes,
            shuffle=shuffle,
        )
        if index >= len(dag_tasks):
            raise IndexError(f"Cannot find task index {index}, only {len(dag_tasks)} tasks matched the filter")
        return dag_tasks[index]

    def build_from_huawei(
        self,
        file_path: str,
        dag_num: Optional[int] = None,
        target_total_nodes: Optional[int] = None,
        min_total_nodes: Optional[int] = None,
        max_total_nodes: Optional[int] = None,
        shuffle: bool = True,
    ) -> List[DatasetDAGTask]:
        grouped_rows: Dict[int, List[dict]] = defaultdict(list)
        with open(file_path, "r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                grouped_rows[int(row["JobId"])].append(row)

        job_items = list(grouped_rows.items())
        if shuffle:
            self.random.shuffle(job_items)

        dag_tasks: List[DatasetDAGTask] = []
        for job_id, rows in job_items:
            total_nodes = len(rows) + 2
            if not self._match_node_filter(total_nodes, target_total_nodes, min_total_nodes, max_total_nodes):
                continue
            dag_tasks.append(self._build_huawei_dag(job_id=job_id, rows=rows, idx=len(dag_tasks)))
            if dag_num is not None and len(dag_tasks) >= dag_num:
                break

        return dag_tasks

    def build_from_cluster(
        self,
        file_path: str,
        dag_num: Optional[int] = None,
        target_total_nodes: Optional[int] = None,
        min_total_nodes: Optional[int] = None,
        max_total_nodes: Optional[int] = None,
    ) -> List[DatasetDAGTask]:
        dag_tasks: List[DatasetDAGTask] = []

        with open(file_path, "r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file, fieldnames=self.CLUSTER_FIELDNAMES)
            current_job_name = None
            current_rows: List[dict] = []

            for row in reader:
                job_name = row["job_name"]
                if current_job_name is None:
                    current_job_name = job_name

                if job_name != current_job_name:
                    total_nodes = len(current_rows) + 2
                    if self._match_node_filter(total_nodes, target_total_nodes, min_total_nodes, max_total_nodes):
                        dag_tasks.append(self._build_cluster_dag(job_name=current_job_name, rows=current_rows, idx=len(dag_tasks)))
                        if dag_num is not None and len(dag_tasks) >= dag_num:
                            return dag_tasks
                    current_job_name = job_name
                    current_rows = []

                current_rows.append(row)

            if current_rows:
                total_nodes = len(current_rows) + 2
                if self._match_node_filter(total_nodes, target_total_nodes, min_total_nodes, max_total_nodes):
                    dag_tasks.append(self._build_cluster_dag(job_name=current_job_name, rows=current_rows, idx=len(dag_tasks)))

        if dag_num is None:
            return dag_tasks
        return dag_tasks[:dag_num]

    def _build_huawei_dag(self, job_id: int, rows: Sequence[dict], idx: int) -> DatasetDAGTask:
        sorted_rows = sorted(rows, key=lambda item: int(item["TaskId"]))
        raw_values = [float(row["ComputeDuration"]) for row in sorted_rows]
        min_value, max_value = self._safe_min_max(raw_values)

        task_records = []
        dataset_id_to_inner_id = {}
        for inner_id, row in enumerate(sorted_rows):
            dataset_task_id = int(row["TaskId"])
            dataset_id_to_inner_id[dataset_task_id] = inner_id
            task_records.append(
                {
                    "inner_id": inner_id,
                    "dataset_task_id": dataset_task_id,
                    "node_value": float(row["ComputeDuration"]),
                    "node_model_key": row["CategoryId"],
                    "raw_row": row,
                }
            )

        dependencies: List[Tuple[int, int]] = []
        edge_weight_map: Dict[Tuple[int, int], float] = {}
        for row in sorted_rows:
            source_inner_id = dataset_id_to_inner_id[int(row["TaskId"])]
            child_tasks = self._parse_python_list(row["ChildTasks"])
            for child_info in child_tasks:
                if not child_info:
                    continue
                child_dataset_id = int(child_info[0])
                if child_dataset_id not in dataset_id_to_inner_id:
                    continue
                target_inner_id = dataset_id_to_inner_id[child_dataset_id]
                dependencies.append((source_inner_id, target_inner_id))
                edge_weight_map[(source_inner_id, target_inner_id)] = float(child_info[1]) if len(child_info) > 1 else 1.0

        return self._materialize_dag(
            idx=idx,
            dataset_name="huawei",
            job_identifier=job_id,
            task_records=task_records,
            dependencies=dependencies,
            raw_value_range=(min_value, max_value),
            edge_weight_map=edge_weight_map,
        )

    def _build_cluster_dag(self, job_name: str, rows: Sequence[dict], idx: int) -> DatasetDAGTask:
        task_rows = list(rows)
        raw_values = [self._cluster_node_value(row) for row in task_rows]
        min_value, max_value = self._safe_min_max(raw_values)

        task_records = []
        original_task_to_inner_id = {}

        for inner_id, row in enumerate(task_rows):
            original_task_to_inner_id[row["task_name"]] = inner_id
            task_records.append(
                {
                    "inner_id": inner_id,
                    "dataset_task_id": row["task_name"],
                    "node_value": self._cluster_node_value(row),
                    "node_model_key": row["task_type"],
                    "raw_row": row,
                }
            )

        dependencies: List[Tuple[int, int]] = []
        for row in task_rows:
            target_inner_id = original_task_to_inner_id[row["task_name"]]
            _, parent_task_numbers = self._parse_cluster_task_name(row["task_name"])
            for parent_task_number in parent_task_numbers:
                parent_task_name = self._resolve_cluster_parent_name(parent_task_number, task_rows)
                if parent_task_name is None:
                    continue
                source_inner_id = original_task_to_inner_id[parent_task_name]
                dependencies.append((source_inner_id, target_inner_id))

        return self._materialize_dag(
            idx=idx,
            dataset_name="cluster",
            job_identifier=job_name,
            task_records=task_records,
            dependencies=dependencies,
            raw_value_range=(min_value, max_value),
            edge_weight_map=None,
        )

    def _materialize_dag(
        self,
        idx: int,
        dataset_name: str,
        job_identifier,
        task_records: Sequence[dict],
        dependencies: Sequence[Tuple[int, int]],
        raw_value_range: Tuple[float, float],
        edge_weight_map: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> DatasetDAGTask:
        total_nodes = len(task_records) + 2
        dag = DatasetDAGTask(total_nodes)
        dag.idx = idx
        dag.info = {
            "dataset": dataset_name,
            "job_id": job_identifier,
            "task_num": len(task_records),
        }

        dag.nodes = [DatasetNode(dag, "start")] + [DatasetNode(dag, record["inner_id"]) for record in task_records] + [DatasetNode(dag, "exit")]
        start_node = dag.nodes[0]
        exit_node = dag.nodes[-1]
        inner_nodes = dag.nodes[1:-1]

        node_lookup = {node.idx: node for node in inner_nodes}
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)

        unique_dependencies = []
        seen_dependencies = set()
        for source_id, target_id in dependencies:
            if source_id == target_id:
                continue
            if (source_id, target_id) in seen_dependencies:
                continue
            seen_dependencies.add((source_id, target_id))
            unique_dependencies.append((source_id, target_id))
            in_degree[target_id] += 1
            out_degree[source_id] += 1

        for record, node in zip(task_records, inner_nodes):
            node.info = record["raw_row"]

        dag.edges = []
        for source_id, target_id in unique_dependencies:
            source_node = node_lookup[source_id]
            target_node = node_lookup[target_id]
            edge = DatasetEdge(source_node, target_node)
            dag.edges.append(edge)
            source_node.out_edges.append(edge)
            target_node.in_edges.append(edge)

        for node in inner_nodes:
            if not node.in_edges:
                edge = DatasetEdge(start_node, node)
                dag.edges.append(edge)
                start_node.out_edges.append(edge)
                node.in_edges.append(edge)
            if not node.out_edges:
                edge = DatasetEdge(node, exit_node)
                dag.edges.append(edge)
                node.out_edges.append(edge)
                exit_node.in_edges.append(edge)

        for node in dag.nodes:
            node.data_size = random.choice(cn.data_size_set)
            node.model_type = random.choice(cn.model_type)
            if node.idx != "start":
                for edge in node.in_edges:
                    edge.weight = node.data_size / len(node.in_edges)

        dag.max_out = max((len(node.out_edges) for node in inner_nodes), default=0)

        graph_nx = nx.DiGraph()
        graph_nx.add_nodes_from([node.idx for node in dag.nodes])
        graph_nx.add_edges_from([edge.to_nx() for edge in dag.edges])
        dag.nx_obj = graph_nx
        dag.adj_matrix_np = nx.to_numpy_array(graph_nx, nodelist=[node.idx for node in dag.nodes], dtype=float, weight="weight")

        return dag

    def _cluster_node_value(self, row: dict) -> float:
        start_time = float(row["start_time"])
        end_time = float(row["end_time"])
        duration = max(end_time - start_time, 1.0)
        instance_num = max(float(row["instance_num"]), 1.0)
        plan_cpu = max(float(row["plan_cpu"]), 1.0)
        plan_mem = max(float(row["plan_mem"]), 0.1)
        return duration * instance_num * plan_cpu * plan_mem

    def _parse_cluster_task_name(self, task_name: str) -> Tuple[Optional[int], List[int]]:
        match = self.CLUSTER_DAG_PATTERN.match(task_name)
        if not match:
            return None, []
        numbers = [int(part) for part in match.group(1).split("_")]
        return numbers[0], numbers[1:]

    def _resolve_cluster_parent_name(self, parent_task_number: int, rows: Sequence[dict]) -> Optional[str]:
        for row in rows:
            task_number, _ = self._parse_cluster_task_name(row["task_name"])
            if task_number == parent_task_number:
                return row["task_name"]
        return None

    def _parse_python_list(self, raw_value: str):
        if raw_value is None or raw_value == "":
            return []
        return ast.literal_eval(raw_value)

    def _map_model_type(self, raw_value) -> int:
        model_choices = list(cn.model_type)
        model_num = len(model_choices)
        if model_num == 0:
            raise ValueError("constants.model_type is empty")

        try:
            numeric_value = int(raw_value)
        except (TypeError, ValueError):
            numeric_value = sum(ord(ch) for ch in str(raw_value))

        return model_choices[numeric_value % model_num]

    def _scale_to_cn_data_size(self, value: float, min_value: float, max_value: float) -> float:
        lower = float(min(cn.data_size_set))
        upper = float(max(cn.data_size_set))
        if max_value <= min_value:
            return (lower + upper) / 2.0
        normalized = (float(value) - min_value) / (max_value - min_value)
        normalized = min(max(normalized, 0.0), 1.0)
        return lower + normalized * (upper - lower)

    def _safe_min_max(self, values: Iterable[float]) -> Tuple[float, float]:
        values = list(values)
        if not values:
            return 0.0, 1.0
        return float(min(values)), float(max(values))

    def _match_node_filter(
        self,
        total_nodes: int,
        target_total_nodes: Optional[int],
        min_total_nodes: Optional[int],
        max_total_nodes: Optional[int],
    ) -> bool:
        if target_total_nodes is not None and total_nodes != target_total_nodes:
            return False
        if min_total_nodes is not None and total_nodes < min_total_nodes:
            return False
        if max_total_nodes is not None and total_nodes > max_total_nodes:
            return False
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build dataset-driven DAG tasks")
    parser.add_argument("--dataset", choices=["huawei", "cluster"], default="huawei")
    parser.add_argument("--file", default="/mnt/storage/hezx/HADES/dataset/Huawei-Network-AI-Challenge/task_table.csv", help="dataset file path")
    parser.add_argument("--dag-num", type=int, default=5)
    parser.add_argument("--target-total-nodes", type=int, default=5)
    parser.add_argument("--min-total-nodes", type=int, default=None)
    parser.add_argument("--max-total-nodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    builder = DatasetDAGTaskBuilder(seed=args.seed)
    dags = builder.build(
        dataset_name=args.dataset,
        file_path=args.file,
        dag_num=args.dag_num,
        target_total_nodes=args.target_total_nodes,
        min_total_nodes=args.min_total_nodes,
        max_total_nodes=args.max_total_nodes,
    )

    print(f"loaded {len(dags)} DAG tasks from {args.dataset}")
    if dags:
        sample_dag = dags[0]
        print(f"sample dag idx={sample_dag.idx}, total_nodes={len(sample_dag.nodes)}, edges={len(sample_dag.edges)}, info={sample_dag.info}")
