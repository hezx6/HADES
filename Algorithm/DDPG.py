import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import copy


# ================= Actor & Critic 网络定义 =================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.net = nn.Sequential(nn.Linear(state_dim, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, action_dim), nn.Sigmoid())

    def forward(self, state):
        action = self.net(state)
        return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, state, action):
        q_value = self.net(torch.cat([state, action], 1))
        return q_value


# ================= OU噪声实现 =================
class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self):
        dx = self.theta * (self.mu - self.state)
        dx += self.sigma * np.random.randn(len(self.state))
        self.state += dx
        return self.state


# ================= DDPG 算法 =================
class DDPG_Agent:
    def __init__(self, state_dim, action_dim, gamma, tau, actor_lr, critic_lr, init_noise_std, device):
        # 初始化网络
        self.device = device
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.net.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.net.parameters(), lr=critic_lr)

        # 超参数
        self.gamma = gamma
        self.tau = tau

        self.replay_buffer = deque(maxlen=1_000_000)

        self.noise = OUNoise(size=action_dim, sigma=init_noise_std)

    # @torch.inference_mode()  # NOTE: runtime optimization
    def select_action(self, state, add_noise=True):
        with torch.no_grad():
            state = torch.as_tensor(state, dtype=torch.float32)
            action = self.actor(state)
            if add_noise:
                noise = torch.as_tensor(self.noise.sample(), dtype=torch.float32).to(self.device)
                action = action + noise
            action = torch.clamp(action, min=0, max=1)
        return action

    def train(self, batch_size=256):
        # 从经验池采样
        states, actions, rewards, next_states, dones = self.sample_batch(batch_size)
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.as_tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(-1)
        next_states = torch.stack(next_states)
        dones = torch.as_tensor(dones, dtype=torch.int8).to(self.device)
        # 计算目标Q值（结合DSR公式）
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_Q = self.critic_target(next_states, next_actions)
            target_Q = rewards + self.gamma * target_Q  # (1 - dones) * self.gamma * target_Q

        # 更新Critic
        current_Q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新目标网络
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)

    def soft_update(self, target, online):
        for t, o in zip(target.parameters(), online.parameters()):
            t.data.copy_(self.tau * o.data + (1 - self.tau) * t.data)

    def sample_batch(self, batch_size):
        trajectorys = random.sample(self.replay_buffer, batch_size)
        batch = list(zip(*trajectorys))
        return [x for x in batch]

    def decay_action_std(self, decay_rate, min_sigma):
        """Decay the action noise standard deviation"""
        self.noise.sigma = max(min_sigma, self.noise.sigma * decay_rate)

    def save_model(self, dir, remark=None):
        if remark is not None:
            torch.save(self.actor.state_dict(), f"{dir}/actor_{remark}.pt")
        else:
            torch.save(self.actor.state_dict(), f"{dir}/actor.pt")

    def load_model(self, dir, remark=None):
        if remark is not None:
            self.actor.load_state_dict(torch.load(f"{dir}/actor_{remark}.pt"))
        else:
            self.actor.load_state_dict(torch.load(f"{dir}/actor.pt"))
