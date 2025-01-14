import gym
import torch
import random
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from torch import nn

device = torch.device( 'cpu')

# 定义卷积神经网络（CNN）模型
class DQNModel(nn.Module):
    def __init__(self, n_actions):
        super(DQNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(4, 16, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(16, 64, kernel_size=2, stride=1)
        self.fc1 = nn.Linear(64 * 1 * 1, 16)
        self.fc2 = nn.Linear(16, n_actions)

    def forward(self, x):
        # 第一层卷积
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        # 第二层卷积
        x = F.relu(F.max_pool2d(self.conv2(x), 2))

        # 第三层卷积
        x = F.relu((self.conv3(x)))

        # 展平
        x = x.view(x.size(0), -1)  # Flatten

        # 第一层全连接
        x = F.relu(self.fc1(x))

        # 输出层
        x = self.fc2(x)
        return x


class DQNAgent:
    def __init__(self, env, model, target_model, optimizer, gamma=0.99, epsilon=1.0, epsilon_decay=0.9999, min_epsilon=0.02):
        self.env = env
        self.model = model
        self.target_model = target_model
        self.optimizer = optimizer
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 64
        self.best_model = None
        self.best_score = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()  # 随机选择动作
        else:
            with torch.no_grad():
                q_values = self.model(state)
                return torch.argmax(q_values, dim=1).item()

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def sample_experience(self):
        return random.sample(self.replay_buffer, self.batch_size)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return  # 如果回放池的样本不足，跳过训练

        experiences = self.sample_experience()
        states, actions, rewards, next_states, dones = zip(*experiences)

        # 转换为 Tensor，并移到正确的设备
        states = torch.cat(states).to(device)
        next_states = torch.cat(next_states).to(device)
        actions = torch.tensor(actions).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards).unsqueeze(1).to(device)
        dones = torch.tensor(dones).unsqueeze(1).to(device)

        # 在此部分添加原地不动的惩罚
        for i in range(len(states)):
            if torch.equal(states[i], next_states[i]):  # 如果状态没有变化，说明模型原地不动
                rewards[i] -= 0.2  # 给原地不动的奖励加上惩罚

        # 计算 Q 值
        q_values = self.model(states).gather(1, actions)  # 当前 Q 网络的 Q 值
        next_q_values = self.target_model(next_states).max(1, keepdim=True)[0]  # 目标网络的 Q 值
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones.int()))  # 目标 Q 值

        # 计算损失
        loss = F.mse_loss(q_values, target_q_values)

        # 更新模型
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新 epsilon
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

