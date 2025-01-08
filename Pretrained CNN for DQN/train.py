import gym
from utils import TrainingLogger, PreprocessState
from DQN_using_resnet import DQNAgentWithPretrained, DQNModelWithPretrained
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import numpy as np
import random
from collections import deque
import logging
import time
import os

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda")

def train(env, model, target_model, logger, num_episodes=100):
    agent = DQNAgentWithPretrained(env, model, target_model, optimizer)
    preprocessor = PreprocessState()
    
    # 确保模型在正确的设备上
    model = model.to(device)
    target_model = target_model.to(device)

    for episode in range(num_episodes):
        state = env.reset()
        processed_state = preprocessor(state).to(device)  # 将预处理后的输入移到GPU
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(processed_state)
            next_state, reward, done, _, info = env.step(action)

            processed_next_state = preprocessor(next_state).to(device)  # 同样地将下一个状态也移到GPU

            agent.store_experience(processed_state, action, reward, processed_next_state, done)
            agent.train()

            processed_state = processed_next_state
            total_reward += reward

        if episode % 10 == 0:
            agent.update_target_model()

        logger.log(episode, total_reward)



if __name__ == "__main__":
    print("Start running!")
    env = gym.make('Breakout-v4', render_mode='rgb_array')
    
    n_actions = env.action_space.n
    model = DQNModelWithPretrained(n_actions).to(device)
    target_model = DQNModelWithPretrained(n_actions).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    log_file = 'training_log.txt'
    logger = TrainingLogger(log_file, model=model, log_interval=1, metric='score')

    logger.start_training()
    train(env, model, target_model, logger)