import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import copy
import torch.nn.functional as F


# ================= 网络定义 =================
class GaussianPolicy(nn.Module):
    """SAC策略网络，输出高斯分布的均值和标准差"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # 限制标准差范围
        return mean, log_std

    def sample(self, state):
        # 重参数化采样
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()  # 重参数化
        action = torch.tanh(z)  # 压缩到[-1,1]
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)  # 修正概率
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob


class TwinQNetwork(nn.Module):
    """双Q网络，输出两个独立的Q值估计"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.q1 = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.q2 = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q1(x), self.q2(x)


# ================= SAC 算法 =================
class SAC_Agent:

    def __init__(self, state_dim, action_dim, gamma, tau, alpha, actor_lr, critic_lr, alpha_lr, target_entropy, device, writer):
        self.device = device
        self.gamma = gamma
        self.tau = tau

        # 初始化策略网络和Q网络
        self.actor = GaussianPolicy(state_dim, action_dim).to(device)
        self.critic = TwinQNetwork(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        # 优化器
        self.policy_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 自动调整温度参数alpha
        self.alpha = alpha
        self.target_entropy = -action_dim if target_entropy is None else target_entropy
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=alpha_lr)

        self.replay_buffer = deque(maxlen=1_000_000)

        self.writer = writer
        self.update_num = 0

    # @torch.inference_mode()  # NOTE: runtime optimization
    def select_action(self, state, deterministic=False):
        # 选择动作（训练时随机采样，评估时取均值）
        with torch.no_grad():  # 重要！！
            state = torch.as_tensor(state).unsqueeze(0).to(self.device)
            if deterministic:
                mean, _ = self.actor(state)
                return torch.sigmoid(mean)
            else:
                action, _ = self.actor.sample(state)
                return action

    def train(self, batch_size):
        # 从经验池采样
        states, actions, rewards, next_states, dones = self.sample_batch(batch_size)
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.stack(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # ----------------- 更新Q网络 -----------------
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            q_target = torch.min(q1_target, q2_target)
            target = rewards + (1 - dones) * self.gamma * (q_target - self.alpha * next_log_probs)

        # Q网络损失
        q1, q2 = self.critic(states, actions)
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.q_optim.zero_grad()
        q_loss.backward()
        self.q_optim.step()

        # ----------------- 更新策略网络 -----------------
        new_actions, log_probs = self.actor.sample(states)
        q1_policy, q2_policy = self.critic(states, new_actions)
        q_policy = torch.min(q1_policy, q2_policy)

        policy_loss = (self.alpha * log_probs - q_policy).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # ----------------- 更新温度参数alpha -----------------
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        # ----------------- 软更新目标网络 -----------------
        self.soft_update(self.critic_target, self.critic)

        if self.update_num % 20 == 0:
            self.writer.add_scalar("algorithm/critic_loss", q_loss.mean().item(), global_step=self.update_num)
            self.writer.add_scalar("algorithm/actor_loss", policy_loss.mean().item(), global_step=self.update_num)
            self.writer.add_scalar("algorithm/alpha_loss", alpha_loss.mean().item(), global_step=self.update_num)
        self.update_num += 1

    def soft_update(self, target, online):
        for t_param, o_param in zip(target.parameters(), online.parameters()):
            t_param.data.copy_(self.tau * o_param.data + (1 - self.tau) * t_param.data)

    def sample_batch(self, batch_size):
        transitions = random.sample(self.replay_buffer, batch_size)
        batch = list(zip(*transitions))
        return [item for item in batch]  # item 已经是tensor

    def save_model(self, dir, remark=None):
        if remark is not None:
            torch.save(self.actor.state_dict(), f"{dir}/actor_{remark}.pt")
            torch.save(self.log_alpha.item(), f"{dir}/alpha_{remark}.txt")
        else:
            torch.save(self.actor.state_dict(), f"{dir}/actor.pt")
            torch.save(self.log_alpha.item(), f"{dir}/alpha.txt")

    def load_model(self, dir, remark=None):
        if remark is not None:
            self.actor.load_state_dict(torch.load(f"{dir}/actor_{remark}.pt"))
            self.log_alpha.data.fill_(torch.load(f"{dir}/alpha_{remark}.txt"))
        else:
            self.actor.load_state_dict(torch.load(f"{dir}/actor.pt"))
            self.log_alpha.data.fill_(torch.load(f"{dir}/alpha.txt"))
