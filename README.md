Source code of paper ``HADES: Heterogeneity-Aware Dependent Task Offloading in Collaborative Edge Networks"


## 运行环境

推荐环境如下：

- OS：Linux
- Python：3.10.12
- Torch：2.2.2
- CUDA：13.0


### Huawei 数据集

默认读取路径：

- [dataset/Huawei-Network-AI-Challenge](dataset/Huawei-Network-AI-Challenge)

### Alibaba 数据集

默认读取路径：

- [dataset/cluster-trace-v2018](dataset/cluster-trace-v2018)

注意：由于`cluster` 数据集中的 `batch_task.csv` 文件体积较大，需要手动按照数据集地址下载文件

## 示例

一个基于 Huawei-Network-AI-Challenge 数据集训练 `HADES` 的示例命令：

```bash
python main.py \
	--mode train \
	--algorithm HADES \
	--dag_source huawei \
	--slot_length 0.2 \
	--num_episode 100 \
	--max_ep_len 128 \
	--batch_size 1024 \
	--es_num 15 \
	--dag_num 5 \
	--dag_node_num 5 \
	--gcn_layer_num 2 \
	--remark demo_huawei \
	-d cuda:0
```