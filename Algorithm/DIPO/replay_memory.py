import numpy as np
import torch


class ReplayMemory:
    """Buffer to store environment transitions."""

    def __init__(self, state_dim, action_dim, capacity, device):
        self.capacity = int(capacity)
        self.device = device

        self.states = np.empty((self.capacity, int(state_dim)), dtype=np.float32)
        self.actions = np.empty((self.capacity, int(action_dim)), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.next_states = np.empty((self.capacity, int(state_dim)), dtype=np.float32)
        self.masks = np.empty((self.capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def append(self, state, action, reward, next_state, mask):

        np.copyto(self.states[self.idx], state)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_states[self.idx], next_state)
        np.copyto(self.masks[self.idx], mask)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=batch_size)

        states = torch.as_tensor(self.states[idxs], device=self.device)
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_states = torch.as_tensor(self.next_states[idxs], device=self.device)
        masks = torch.as_tensor(self.masks[idxs], device=self.device)

        return states, actions, rewards, next_states, masks


class list_ReplayMemory:
    """Buffer to store environment transitions."""

    def __init__(self, capacity, device):
        self.capacity = int(capacity)
        self.device = device

        # 使用列表存储数据
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []

        self.idx = 0
        self.full = False

    def append(self, state, action, reward, next_state):
        # 如果列表未满，直接添加数据
        if len(self.states) < self.capacity:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_states.append(next_state)
        else:
            # 如果列表已满，替换最旧的数据
            self.states[self.idx] = state
            self.actions[self.idx] = action
            self.rewards[self.idx] = reward
            self.next_states[self.idx] = next_state

        # 更新索引和 full 标志
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        """Sample a batch of transitions from memory."""
        # 确定可用的索引范围
        max_idx = self.capacity if self.full else self.idx
        idxs = np.random.randint(0, max_idx, size=batch_size)

        # 从列表中按索引提取数据
        states = [self.states[i] for i in idxs]
        actions = [self.actions[i] for i in idxs]
        rewards = [self.rewards[i] for i in idxs]
        next_states = [self.next_states[i] for i in idxs]

        return states, actions, rewards, next_states


class DiffusionMemory:
    """Buffer to store best actions."""

    def __init__(self, capacity, device):
        self.capacity = int(capacity)
        self.device = device

        # 使用列表存储数据
        self.states = []
        self.best_actions = []

        self.idx = 0
        self.full = False

    def append(self, state, action):
        # 如果列表未满，直接添加数据
        if len(self.states) < self.capacity:
            self.states.append(state)
            self.best_actions.append(action)
        else:
            # 如果列表已满，替换最旧的数据
            self.states[self.idx] = state
            self.best_actions[self.idx] = action

        # 更新索引和 full 标志
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size, squeeze=False):
        """Sample a batch of transitions from memory."""
        # 确定可用的索引范围
        max_idx = self.capacity if self.full else self.idx
        idxs = np.random.randint(0, max_idx, size=batch_size)

        # 从列表中按索引提取数据并转换为 PyTorch 张量
        states = [self.states[i] for i in idxs]
        best_actions = torch.stack([self.best_actions[i] for i in idxs]).to(self.device)
        if squeeze:
            best_actions = best_actions.squeeze(1)
        # 设置 best_actions 的 requires_grad 为 True

        best_actions.requires_grad_(True)
        return states, best_actions, idxs

    def replace(self, idxs, best_actions, squeeze=False):
        """Replace best actions at specific indices."""
        # 将更新后的 best_actions 赋值给列表中的相应位置
        if squeeze:
            best_actions = best_actions.unsqueeze(1)
        for i, idx in enumerate(idxs):
            self.best_actions[idx] = best_actions[i]
