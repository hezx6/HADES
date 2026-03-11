import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from typing import Dict
from torch.distributions import Normal, Independent
import numpy as np
import os
from tensorboardX import SummaryWriter

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,7,8"


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def append(self, state, action, reward, done, action_logprob, state_val):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(action_logprob)
        self.state_values.append(state_val)
        self.rewards.append(reward)
        self.is_terminals.append(done)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init)

        self.action_head = nn.Linear(64, self.action_dim)

        # critic
        self.critic = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def set_action_std(self, new_action_std, device):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def forward(self, x: torch.Tensor):

        x = self.encoder(x)
        # 输出每个动作的均值
        logit = self.action_head(x)

        return logit

    def act(self, state, policy, device):
        """
        输入 logit, 采样得到动作
        """
        logit = policy(state)

        action_mean1 = torch.sigmoid(logit)
        # print('action_mean1:',action_mean1)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0).to(device)
        # print('cov_mat:',cov_mat)
        dist = MultivariateNormal(action_mean1, cov_mat)
        action = dist.sample()
        # print('action:',action)
        action_logprob = dist.log_prob(action)
        # print('action_logprob1:',action_logprob1)
        # raise Exception("stop!")

        state_val = self.critic(state)

        return action, action_logprob.detach(), state_val.detach()

    def evaluate(self, policy, state, action, device):

        logit = policy(state)

        action_mean = torch.sigmoid(logit)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)  # diag_embed:将指定数组变成对角阵
        dist = MultivariateNormal(action_mean, cov_mat)
        action = action.reshape(-1, self.action_dim)  # action_dim=1，take this opration！
        action_logprob = dist.log_prob(action)
        action_dist_entropy = dist.entropy()

        state_val = self.critic(state)

        return action_logprob, action_dist_entropy, state_val


class PPO_Agent_c:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, coef_entropy, action_std_init, device, writer):

        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.coef_entropy = coef_entropy
        self.action_std = action_std_init

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, self.action_std).to(self.device)
        self.optimizer = torch.optim.Adam([{"params": self.policy.encoder.parameters(), "lr": lr_actor}, {"params": self.policy.critic.parameters(), "lr": lr_critic}])

        self.policy_old = ActorCritic(state_dim, action_dim, self.action_std).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.writer = writer
        self.update_num = 0

    def append_memory(self, state, action, reward, done, action_logprob, state_val):
        self.buffer.append(state, action, reward, done, action_logprob, state_val)

    def select_action(self, state):
        with torch.no_grad():
            action, action_logprob, state_val = self.policy_old.act(state, self.policy_old, self.device)
            action = torch.clamp(action[0], 0, 1)

        return action, action_logprob, state_val

    def set_action_std(self, new_action_std, device):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std, device)
        self.policy_old.set_action_std(new_action_std, device)

    def decay_action_std(self, action_std_decay_rate, min_action_std):

        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if self.action_std <= min_action_std:
            self.action_std = min_action_std
            # print("setting actor output action_std to min_action_std : ", self.action_std)
        self.set_action_std(self.action_std, self.device)

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
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_action_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)

        # calculate advantages（GAE）
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            action_logprob, action_entropy, state_values = self.policy.evaluate(self.policy, old_states, old_actions, self.device)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(action_logprob - old_action_logprobs.detach())
            # ratios2 = torch.exp(action_args_logprob.sum(1, keepdim=True) - old_action_args_logprobs.sum(1, keepdim=True))

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - self.coef_entropy * action_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            self.writer.add_scalar("algorithm/ppo_loss", loss.mean().item(), global_step=self.update_num)

        self.update_num += 1
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

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
