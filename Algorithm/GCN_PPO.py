import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch.distributions import MultivariateNormal
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import numpy as np
import random

torch.autograd.set_detect_anomaly(True)


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.state_values.clear()
        self.is_terminals.clear()


class GCN_Critic(nn.Module):

    def __init__(self, net_node_feat_dim, dag_node_feat_dim, dag_embedding_dim, net_embedding_dim, normalize):
        super(GCN_Critic, self).__init__()

        self.normalize = normalize
        # GCN for network and DAG
        self.net_encoder = GCNConv(net_node_feat_dim, net_embedding_dim)
        self.dag_encoder = GCNConv(dag_node_feat_dim, dag_embedding_dim)

        # Concatenation and processing
        self.concate_encoder = nn.Sequential(nn.Linear(dag_embedding_dim + net_embedding_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, net_feat, net_edge_index, net_edge_weights, dag_feat, dag_edge_index, dag_edge_weights, net_batch=None, dag_batch=None):

        # Apply GCN to network and DAG features
        net_embedding = self.net_encoder(net_feat, net_edge_index, edge_weight=net_edge_weights)
        dag_embedding = self.dag_encoder(dag_feat, dag_edge_index, edge_weight=dag_edge_weights)

        net_embedding = F.relu(net_embedding)
        dag_embedding = F.relu(dag_embedding)

        if self.normalize == True:
            net_embedding = torch.nn.functional.normalize(net_embedding, dim=-1)
            dag_embedding = torch.nn.functional.normalize(dag_embedding, dim=-1)

        if net_batch != None and dag_batch != None:  # batch gru processing when updating

            net_embedding = global_mean_pool(net_embedding, net_batch)  # mean pooling based on batch
            dag_embedding = global_mean_pool(dag_embedding, dag_batch)  # mean pooling based on batch

            combined_embedding = torch.cat((net_embedding, dag_embedding), dim=-1)  # [batch_size,embedding_dim]
            state_vals = self.concate_encoder(combined_embedding)
            return state_vals.squeeze(1)
        else:
            # Assuming we concatenate embeddings by their mean or max
            net_embedding = net_embedding.mean(dim=0)
            dag_embedding = dag_embedding.mean(dim=0)

            # Concatenate and apply fully connected layers
            combined_embedding = torch.cat((net_embedding, dag_embedding), dim=-1)
            state_val = self.concate_encoder(combined_embedding)

            return state_val


class GCN_Actor(nn.Module):

    def __init__(self, net_node_feat_dim, dag_node_feat_dim, dag_embedding_dim, net_embedding_dim, num_net_gcn_layers, num_dag_gcn_layers, head_in_dim, action_len, action_dim, action_std_init, is_attention, normalize):
        super(GCN_Actor, self).__init__()

        self.normalize = normalize
        self.net_embedding_dim = net_embedding_dim
        self.dag_embedding_dim = dag_embedding_dim
        self.net_gcn_layers = nn.ModuleList([GCNConv(net_node_feat_dim if i == 0 else net_embedding_dim, net_embedding_dim) for i in range(num_net_gcn_layers)])

        self.dag_gcn_layers = nn.ModuleList([GCNConv(dag_node_feat_dim if i == 0 else dag_embedding_dim, dag_embedding_dim) for i in range(num_dag_gcn_layers)])

        self.is_attention = is_attention
        if is_attention == True:
            self.W_a = torch.nn.Parameter(torch.FloatTensor(dag_embedding_dim * 2, 1))
            torch.nn.init.kaiming_uniform_(self.W_a)

        self.leakyrelu = nn.LeakyReLU(0.01)

        self.action_head = nn.Sequential(nn.Linear(head_in_dim, 256), nn.ReLU(), nn.Linear(256, action_len * action_dim))

        self.action_dim = action_dim
        self.dag_nodes_num = action_len
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init)

    def set_action_std(self, new_action_std, device):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def forward(self, net_feat, net_edge_index, net_edge_weights, dag_feat, dag_edge_index, dag_edge_weights, net_batch=None, dag_batch=None):
        # net_feat: 节点特征矩阵, shape = [num_nodes, num_node_features]
        # edge_index: 边的索引矩阵, shape = [2, num_edges],第一行为边的起点索引，第二行为边的终点索引，每一列中的两个元素表示一条边

        # ***如果输入的数据是torch_geometric.data.Batch封装过的数据（Batch.from_data_list(graph_list)），经过GCNConv之后，embedding的shape已经变成了[batch,embedding_dim]，后续可以直接通过embedding[graph_idx]来得到第idx个图的embedding***
        for i, gcn in enumerate(self.net_gcn_layers):
            net_embedding = F.relu(gcn(net_feat, net_edge_index, edge_weight=net_edge_weights)) if i == 0 else F.relu(gcn(net_embedding, net_edge_index, edge_weight=net_edge_weights))

        for i, gcn in enumerate(self.dag_gcn_layers):
            dag_embedding = F.relu(gcn(dag_feat, dag_edge_index, edge_weight=dag_edge_weights)) if i == 0 else F.relu(gcn(dag_embedding, dag_edge_index, edge_weight=dag_edge_weights))

        if self.normalize == True:
            net_embedding = torch.nn.functional.normalize(net_embedding, dim=-1)
            dag_embedding = torch.nn.functional.normalize(dag_embedding, dim=-1)

        if net_batch != None and dag_batch != None:  # batch processing when updating

            batch_size = net_batch.max().item() + 1  # Get the number of nodes per dag graph

            net_embedding = net_embedding.reshape(batch_size, -1, self.net_embedding_dim)
            dag_embedding = dag_embedding.reshape(batch_size, -1, self.dag_embedding_dim)

            if self.is_attention == True:
                # Attention_layer
                # (batch_size, net_node_num, embedding) 与 (batch_size, dag_nodes_num, embedding) 进行拼接
                net_embedding_expanded = net_embedding.unsqueeze(2).expand(-1, -1, self.dag_nodes_num, -1)  # (batch_size, net_node_num, dag_nodes_num, embedding)
                dag_embedding_expanded = dag_embedding.unsqueeze(1).expand(-1, net_embedding.size(1), -1, -1)  # (batch_size, net_node_num, dag_nodes_num, embedding)

                # 拼接并计算得分
                combined = torch.cat((net_embedding_expanded, dag_embedding_expanded), dim=-1)  # (batch_size, net_node_num, dag_nodes_num, 2 * embedding)

                # 计算注意力得分
                alpha_s = self.leakyrelu(torch.matmul(combined, self.W_a).squeeze(-1) + 1e-7)  # [batch_size, net_node_num, dag_nodes_num], W_a 的 shape 应为 (2 * embedding, out_dim)

                assert torch.isnan(alpha_s[0][0, 0]) != True, "出现了nan"
                # 归一化
                softmax_alpha = F.softmax(alpha_s, dim=-1)

                # hyb_embedding_net = net_embedding.clone()  # 保留net_embedding原始特征
                # hyb_embedding_dag = dag_embedding_pad.clone()  # 保留net_embedding原始特征

                hyb_embedding_net = net_embedding + torch.bmm(softmax_alpha, dag_embedding)
                hyb_embedding_dag = dag_embedding + torch.bmm(softmax_alpha.transpose(1, 2), net_embedding)

            else:
                net_node_num = net_embedding.size(1)
                alpha = torch.ones(batch_size, net_node_num, self.dag_nodes_num).to(net_embedding.device)
                tmp1 = torch.matmul(alpha, dag_embedding)
                hyb_embedding_net = net_embedding + tmp1
                hyb_embedding_dag = dag_embedding + torch.matmul(alpha.transpose(1, 2), net_embedding)

            hyb_embedding = (hyb_embedding_net.mean(dim=1) + hyb_embedding_dag.mean(dim=1)) / 2

            actions = self.action_head(hyb_embedding).reshape(batch_size, self.dag_nodes_num, self.action_dim)

            return actions

        else:  # not batch processing

            net_node_num = net_embedding.size(0)
            dag_node_num = dag_embedding.size(0)
            # Attention_layer
            if self.is_attention == True:
                alpha_s = torch.zeros(net_node_num, dag_node_num)

                net_embedding_expanded = net_embedding.unsqueeze(1).repeat(1, dag_node_num, 1)  # [net_emb_num, dag_emb_num, embedding_dim]
                dag_embedding_expanded = dag_embedding.unsqueeze(0).repeat(net_node_num, 1, 1)  # [net_emb_num, dag_emb_num, embedding_dim]

                # 拼接节点特征
                combined = torch.cat([net_embedding_expanded, dag_embedding_expanded], dim=-1)  # [net_emb_num, dag_emb_num, 2 * embedding_dim]

                # 计算注意力得分
                e_ij = self.leakyrelu(torch.matmul(combined, self.W_a).squeeze(-1) + 1e-7)  # [net_emb_num, dag_emb_num]

                assert torch.isnan(e_ij[0, 0]) != True, "出现了nan"
                # 归一化
                alpha_s = F.softmax(e_ij, dim=1)
                # 根据attention系数进行加权和，融合特征
                # hyb_embedding_net = net_embedding.clone()  # 保留net_embedding原始特征
                # hyb_embedding_dag = dag_embedding.clone()  # 保留dag_embedding原始特征

                hyb_embedding_net = net_embedding + torch.matmul(alpha_s, dag_embedding)
                hyb_embedding_dag = dag_embedding + torch.matmul(alpha_s.t(), net_embedding)

            else:  # no attention, all weights between any graph nodes = 1
                alpha = torch.ones(net_node_num, dag_node_num).to(net_embedding.device)  # [dag_node_num, net_node_num]
                hyb_embedding_net = net_embedding + torch.matmul(alpha, dag_embedding) / dag_node_num
                hyb_embedding_dag = dag_embedding + torch.matmul(alpha.t(), net_embedding) / net_node_num

            hyb_embedding = (hyb_embedding_net.mean(dim=0) + hyb_embedding_dag.mean(dim=0)) / 2

            action = self.action_head(hyb_embedding).reshape(self.dag_nodes_num, self.action_dim)

            return action


class GCN_PPO:
    def __init__(
        self,
        net_node_feat_dim,
        dag_node_feat_dim,
        dag_embedding_dim,
        net_embedding_dim,
        num_net_gcn_layers,
        num_dag_gcn_layers,
        head_in_dim,
        action_dim,
        action_len,
        lr_actor,
        lr_critic,
        gamma,
        epochs_num,
        batch_size,
        eps_clip,
        coef_entropy,
        action_std_init,
        is_attention,
        normalize,
        device,
        writer,
    ):

        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.epochs_num = epochs_num
        self.batch_size = batch_size
        self.coef_entropy = coef_entropy
        self.action_dim = action_dim
        self.action_std = action_std_init
        self.is_attention = is_attention

        self.buffer = RolloutBuffer()

        self.actor = GCN_Actor(net_node_feat_dim, dag_node_feat_dim, dag_embedding_dim, net_embedding_dim, num_net_gcn_layers, num_dag_gcn_layers, head_in_dim, action_len, action_dim, action_std_init, is_attention, normalize).to(
            self.device
        )

        self.critic = GCN_Critic(net_node_feat_dim, dag_node_feat_dim, dag_embedding_dim, net_embedding_dim, normalize).to(self.device)

        self.optimizer = torch.optim.Adam([{"params": self.actor.parameters(), "lr": lr_actor}, {"params": self.critic.parameters(), "lr": lr_critic}])

        self.actor_old = GCN_Actor(
            net_node_feat_dim, dag_node_feat_dim, dag_embedding_dim, net_embedding_dim, num_net_gcn_layers, num_dag_gcn_layers, head_in_dim, action_len, action_dim, action_std_init, is_attention, normalize
        ).to(self.device)
        self.actor_old.load_state_dict(self.actor.state_dict())

        self.critic_old = GCN_Critic(net_node_feat_dim, dag_node_feat_dim, dag_embedding_dim, net_embedding_dim, normalize).to(self.device)
        self.critic_old.load_state_dict(self.critic.state_dict())

        self.MseLoss = nn.MSELoss()
        self.writer = writer
        self.update_num = 0

    def act(self, net_feature, net_edge_index, net_edge_weights, dag_feature, dag_edge_index, dag_edge_weights, device):
        """
        输入 logit, 采样得到动作
        """
        logit = self.actor_old(net_feature, net_edge_index, net_edge_weights, dag_feature, dag_edge_index, dag_edge_weights)

        # infer action [Node_num * action_dim]
        action_mean = torch.sigmoid(logit)
        cov_mat1 = torch.diag(self.actor.action_var).unsqueeze(0).to(device)
        dist1 = MultivariateNormal(action_mean, cov_mat1)
        action = dist1.sample()

        action_mean_squeeze = action_mean.view(logit.size(0) * logit.size(1))
        cov_mat_squeeze = torch.diag(torch.full((logit.size(0) * logit.size(1),), self.actor.action_var[0])).unsqueeze(dim=0).to(device)
        dist2 = MultivariateNormal(action_mean_squeeze, cov_mat_squeeze)
        action_logprob = dist2.log_prob(action.view(action.size(0) * action.size(1)))

        state_val = self.critic(net_feature, net_edge_index, net_edge_weights, dag_feature, dag_edge_index, dag_edge_weights)

        return action, action_logprob.detach(), state_val.detach()

    def evaluate(self, policy, state, action, batch_indices):

        net_list = [
            Data(
                x=state[i]["net_feature"],
                edge_index=state[i]["net_edge_index"],
                edge_attr=state[i]["net_edge_weights"],
            )
            for i in batch_indices
        ]
        dag_list = [
            Data(
                x=state[i]["dag_feature"],
                edge_index=state[i]["dag_edge_index"],
                edge_attr=state[i]["dag_edge_weights"],
            )
            for i in batch_indices
        ]

        action = [action[i] for i in batch_indices]

        net_batch = Batch.from_data_list(net_list)
        dag_batch = Batch.from_data_list(dag_list)

        logit_list = policy(net_batch.x, net_batch.edge_index, net_batch.edge_attr, dag_batch.x, dag_batch.edge_index, dag_batch.edge_attr, net_batch.batch, dag_batch.batch)  # batch*node_num*5

        action_logprobs = []
        action_entropies = []

        # 遍历每个 logit 张量
        for i in range(len(logit_list)):
            flat_action_mean = torch.sigmoid(logit_list[i]).view(-1)  # flatten：[Node_num,action_dim] --> [Node_num*action_dim]

            # 计算协方差矩阵
            cov_mat = torch.diag(torch.full((flat_action_mean.size(0),), self.action_std * self.action_std)).unsqueeze(0).to(self.device)  # 1*[(Node_num * action_dim)*(Node_num * action_dim)]

            # 创建多元正态分布
            dist = MultivariateNormal(flat_action_mean, cov_mat)

            # 计算 action 的 log_prob 和 entropy
            action_logprob = dist.log_prob(action[i].view(-1))  # 将第action flatten
            action_entropy = dist.entropy()  # 计算熵

            # 将每个 action 的 log_prob 和 entropy 存储到列表中
            action_logprobs.append(action_logprob)
            action_entropies.append(action_entropy)

        batch_action_logprob = torch.cat(action_logprobs)
        batch_action_dist_entropy = torch.cat(action_entropies)

        # # 每个action的长度不一致，无法用一个高维tensor进行表示，只能采用一个tensor列表，导致计算效率低
        # action_mean = torch.tensor([torch.sigmoid(logit) for logit in logit_list]).view(len(logit_list), -1)  # batch*(N*5)=batch*5N
        # action_var = self.action_var.expand_as(action_mean)  # batch*(12*5)=batch*5N
        # cov_mat = torch.diag_embed(action_var)  # diag_embed:将指定数组变成对角阵, batch*60*60
        # dist = MultivariateNormal(action_mean, cov_mat)  # batch*60
        # # action = action.reshape(-1, self.action_dim)  # action_dim=1，take this opration！
        # action_logprob = dist.log_prob(action.view(action.size(0), action.size(1) * action.size(2)))  # [batch]
        # action_dist_entropy = dist.entropy()

        state_val = self.critic(net_batch.x, net_batch.edge_index, net_batch.edge_attr, dag_batch.x, dag_batch.edge_index, dag_batch.edge_attr, net_batch.batch, dag_batch.batch).squeeze(-1)  # [256]

        return batch_action_logprob, batch_action_dist_entropy, state_val

    def append_memory(self, state, action, reward, done, action_logprob, state_val):
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(done)

    # @torch.inference_mode()  # NOTE: runtime optimization
    def select_action(self, net_feature, net_edge_index, net_edge_weights, dag_feature, dag_edge_index, dag_edge_weights):
        with torch.no_grad():

            net_feature = net_feature
            net_edge_index = net_edge_index
            net_edge_weights = net_edge_weights

            dag_feature = dag_feature
            dag_edge_index = dag_edge_index
            dag_edge_weights = dag_edge_weights

            # =========  根据 logit 采样得到动作  ===============
            logit = self.actor(net_feature, net_edge_index, net_edge_weights, dag_feature, dag_edge_index, dag_edge_weights)

            # infer action [Node_num * action_dim]
            action_mean = torch.sigmoid(logit)
            cov_mat1 = torch.diag(self.actor.action_var).unsqueeze(0).to(self.device)
            dist1 = MultivariateNormal(action_mean, cov_mat1)
            action = dist1.sample()

            # =========  计算动作分布的对数概率  ===============
            action_mean_squeeze = action_mean.view(logit.size(0) * logit.size(1))
            cov_mat_squeeze = torch.diag(torch.full((logit.size(0) * logit.size(1),), self.actor.action_var[0])).unsqueeze(dim=0).to(self.device)
            dist2 = MultivariateNormal(action_mean_squeeze, cov_mat_squeeze)
            action_logprob = dist2.log_prob(action.view(action.size(0) * action.size(1)))

            # =========  计算state val 和 cost val  ===============
            state_val = self.critic(net_feature, net_edge_index, net_edge_weights, dag_feature, dag_edge_index, dag_edge_weights)

            cliped_action = torch.clamp(action, 0.001, 1)

        return cliped_action, action_logprob.detach(), state_val.detach()

    def decay_action_std(self, action_std_decay_rate, min_action_std):

        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if self.action_std <= min_action_std:
            self.action_std = min_action_std

        self.actor.set_action_std(self.action_std, self.device)
        self.actor_old.set_action_std(self.action_std, self.device)

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        detached_old_states = self.buffer.states
        detached_old_actions = self.buffer.actions
        old_action_logprobs = self.buffer.logprobs

        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)

        # calculate advantages（GAE）
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.epochs_num):

            # Evaluating old actions and values
            batch_indices = random.sample(range(len(self.buffer.states)), self.batch_size)
            batch_advantages = torch.index_select(advantages, dim=0, index=torch.tensor(batch_indices).to(self.device))
            batch_rewards = torch.index_select(rewards, dim=0, index=torch.tensor(batch_indices).to(self.device))

            action_logprob, action_entropy, state_values = self.evaluate(self.actor, detached_old_states, detached_old_actions, batch_indices)

            batch_old_action_logprobs = torch.cat([old_action_logprobs[i] for i in batch_indices]).squeeze(-1)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(action_logprob - batch_old_action_logprobs)
            # ratios2 = torch.exp(action_args_logprob.sum(1, keepdim=True) - old_action_args_logprobs.sum(1, keepdim=True))

            # Finding Surrogate Loss
            surr1 = ratios * batch_advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, batch_rewards) - self.coef_entropy * action_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            if self.is_attention == True:
                nn.utils.clip_grad_norm_(self.actor.W_a, max_norm=1.0)
                self.writer.add_scalar("algorithm/w_a0", self.actor.W_a.detach()[0][0], global_step=self.update_num)
                self.writer.add_scalar("algorithm/att_para", self.actor.W_a.grad.norm(), global_step=self.update_num)
            self.optimizer.step()
            self.writer.add_scalar("algorithm/loss", loss.mean().item(), global_step=self.update_num)

        self.update_num += 1
        # Copy new weights into old policy
        self.actor_old.load_state_dict(self.actor.state_dict())

        # clear buffer
        self.buffer.clear()

    def save_model(self, dir, remark=None):
        if remark is not None:
            torch.save(self.actor.state_dict(), f"{dir}/actor_{remark}.pth")
        else:
            torch.save(self.actor.state_dict(), f"{dir}/actor.pth")

    def load_model(self, dir, remark=None):
        if remark is not None:
            self.actor.load_state_dict(torch.load(f"{dir}/actor_{remark}.pth"))
        else:
            self.actor.load_state_dict(torch.load(f"{dir}/actor.pth"))
