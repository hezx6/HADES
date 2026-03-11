## Source code of paper "HADES: Heterogeneity-Aware Dependent Task Offloading in Collaborative Edge Networks"


## Environment

- OS：Linux
- Python：3.10.12
- Torch：2.2.2
- CUDA：13.0


### Huawei Dataset

Default path： [dataset/Huawei-Network-AI-Challenge](dataset/Huawei-Network-AI-Challenge)

### Alibaba Dataset

Default path： [dataset/cluster-trace-v2018](dataset/cluster-trace-v2018)

Note: The `batch_task.csv` file in this dataset is relatively large, so it needs to be downloaded manually from the dataset source.

## Example

An example command for training `HADES` on the Huawei-Network-AI-Challenge dataset:

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