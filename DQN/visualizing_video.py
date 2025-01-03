import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from DQN import DQNModel
from utils import PreprocessState
import random

# 重玩游戏并展示视频
def replay(env, model, num_episodes=1):
    preprocessor = PreprocessState()

    # 确保模型在正确的设备上
    model = model.to(device)
    model.eval()  # 切换到评估模式，关闭梯度计算

    for episode in range(num_episodes):
        state = env.reset()  # 重置环境，返回初始状态
        
        
        processed_state = preprocessor(state).to(device)  # 预处理并将输入移到GPU
        done = False
        total_reward = 0

        while not done:
            # 模型根据当前状态选择动作
            q_values = model(processed_state)
            action = None
            if random.random() < 0.5:
                action = env.action_space.sample()  # 随机选择动作
            else: 
                action = torch.argmax(q_values, dim=1).item()  # 选择最大Q值对应的动作
            # 执行动作并获得下一状态和奖励
            print(action, end='')
            next_state, reward, done, _, info = env.step(action)

            # 预处理下一个状态
            processed_next_state = preprocessor(next_state).to(device)

            total_reward += reward

            # 渲染游戏画面
            env.render()

            processed_state = processed_next_state  # 更新状态

        print(f"Episode {episode + 1}: Total reward = {total_reward}")

# 运行重玩函数
if __name__ == "__main__":
    env = gym.make('Breakout-v4', render_mode='human')  # 创建环境
    device = torch.device("cpu")
    n_actions = env.action_space.n
    model = DQNModel(n_actions)  # 假设你已经有了DQN模型
    model_path = './checkpoints/best_model_v7.pth'  # 训练好的模型路径
    model.load_state_dict(torch.load(model_path))  # 加载权重

    # 重玩游戏并展示视频
    replay(env, model)

