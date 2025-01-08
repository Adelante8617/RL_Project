import gym
import torch
import random
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from torch import nn
import os

device = torch.device("cuda")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DQNModelWithPretrained(nn.Module):
    def __init__(self, n_actions):
        super(DQNModelWithPretrained, self).__init__()

        # 加载预训练的 ResNet18 模型
        resnet = models.resnet18(pretrained=True)

        # 去掉 ResNet18 的最后一层全连接层
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # 只保留卷积层部分

        # 冻结所有层的参数
        for param in self.features.parameters():
            param.requires_grad = False

        # 新增全连接层来适配 DQN 输出
        self.fc1 = nn.Linear(resnet.fc.in_features, 256)  # 这里的特征数量需要根据预训练模型调整
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x):
        # 假设输入图像是大小为 (batch_size, 3, 224, 224)
        x = self.features(x)  # 经过 ResNet 的卷积部分
        x = x.view(x.size(0), -1)  # 展平特征
        x = F.relu(self.fc1(x))  # 第一层全连接
        x = self.fc2(x)  # 输出层
        return x


class DQNAgentWithPretrained:
    def __init__(self, env, model, target_model, optimizer, model_path="./checkpoints/latest_model.pth", gamma=0.99, epsilon=1.0, epsilon_decay=0.9999, min_epsilon=0.02):
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
        self.model_path = model_path  # 类成员变量

        # 加载模型权重（如果存在）
        self.load_model_weights()

    def load_model_weights(self):
        """加载已保存的模型权重"""
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path)
            self.model.load_state_dict(checkpoint)  
            print("Load success")
        else:
            print("未找到已保存的模型权重，开始从头训练")

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
