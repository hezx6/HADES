import argparse
from collections import namedtuple
from itertools import count

import os, sys, random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from tensorboardX import SummaryWriter



class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        s, s_, a, r, d = [], [], [], [], []

        for i in ind:
            S,  A, R, S_, D = self.storage[i]
            s.append(np.array(S, copy=False))
            s_.append(np.array(S_, copy=False))
            a.append(np.array(A, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(s), np.array(s_), np.array(a), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)


    def forward(self, state):
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        a = torch.sigmoid(self.fc3(a)) 
        return a
    
    
class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                # m.weight.data.normal_(mean = 0, std = 0.01)
                # m.bias.data.fill_(0.0)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)

        q = F.relu(self.fc1(state_action))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q


class TD3_Agent():
    def __init__(self, state_dim, action_dim,device,batch_size,policy_noise,noise_clip,gamma,policy_delay,tau,noise_std,max_buffer_size,writer):
        self.batch_size = batch_size
        self.device = device
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.gamma = gamma
        self.policy_delay = policy_delay
        self.tau = tau
        self.max_buffer_size = max_buffer_size
        self.action_dim = action_dim
        self.noise_std = noise_std
        self.actor_pointer = 0

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.critic_1 = Critic(state_dim, action_dim).to(self.device)
        self.critic_1_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_2 = Critic(state_dim, action_dim).to(self.device)
        self.critic_2_target = Critic(state_dim, action_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr = 0.0003)
        self.scheduler_actor = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size= 4000, gamma=0.7)

        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(),lr = 0.0003)
        #self.scheduler_critic1 = torch.optim.lr_scheduler.StepLR(self.critic_1_optimizer, step_size= 8000, gamma=0.7)

        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(),lr = 0.0003)
        #self.scheduler_critic2 = torch.optim.lr_scheduler.StepLR(self.critic_2_optimizer, step_size= 8000, gamma=0.7)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.buffer = Replay_buffer(self.max_buffer_size)
        self.writer = writer
        self.update_num = 0
        input_ = torch.rand(state_dim).to(self.device)
        self.writer.add_graph(self.actor,input_)

    def decay_action_std(self, noise_std_decay_rate, min_noise_std):
        
        self.noise_std = self.noise_std - noise_std_decay_rate
        self.noise_std = round(self.noise_std, 4)
        if (self.noise_std <= min_noise_std):
            self.noise_std = min_noise_std
            print("setting actor output action_std to min_action_std : ", self.noise_std)
        else:
            print("setting actor output action_std to : ", self.noise_std)

    def select_action(self, state):
        if len(self.buffer.storage) < self.batch_size:
            action =  np.random.uniform(0.001,1,self.action_dim)
            return action
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action = self.actor(state)
                action = action.cpu() + np.random.normal(0, self.noise_std, size=self.action_dim)
        #print('action = ',action.cpu().numpy())
        #raise Exception('stop')
            return np.clip(action.numpy(),0,1)
        
    def append_memory(self, state, action, reward, next_state, done):
        self.buffer.push((state, action,  reward, next_state, done))

    def update(self, num_iteration):
        self.actor_pointer += 1
        for _ in range(num_iteration):
            x, y, u, r, d = self.buffer.sample(self.batch_size)
            state = torch.FloatTensor(x).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            next_state = torch.FloatTensor(y).to(self.device)
            done = torch.FloatTensor(d).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)

            # Select next action according to target policy:
            with torch.no_grad():
                noise = torch.ones_like(action).data.normal_(0, self.policy_noise).to(self.device)
                noise = noise.clamp(-self.noise_clip, self.noise_clip)
                next_action = self.actor_target(next_state) + noise

                # Compute target Q-value:
                target_Q1 = self.critic_1_target(next_state, next_action)
                target_Q2 = self.critic_2_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + ((1 - done) * self.gamma * target_Q).detach()

            # Optimize Critic 1:
            current_Q1 = self.critic_1(state, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()
            #self.scheduler_critic1.step()
            if self.update_num % 200 == 0:
                self.writer.add_scalar('gradiant/loss_Q1', loss_Q1, global_step=self.update_num)

            # Optimize Critic 2:
            current_Q2 = self.critic_2(state, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()
            #self.scheduler_critic2.step()
            if self.update_num % 200 == 0:
                self.writer.add_scalar('gradiant/loss_Q2', loss_Q2, global_step=self.update_num)

            # Delayed policy updates:
            if self.actor_pointer % self.policy_delay == 0:
                # Compute actor loss:
                actor_loss = - self.critic_1(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.scheduler_actor.step()
                if self.update_num % 200 == 0:
                    self.writer.add_scalar('gradiant/actor_loss', actor_loss, global_step=self.update_num)


                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(((1- self.tau) * target_param.data) + self.tau * param.data)

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(((1 - self.tau) * target_param.data) + self.tau * param.data)

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(((1 - self.tau) * target_param.data) + self.tau * param.data)
                    
        self.update_num += 1

