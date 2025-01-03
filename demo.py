import gym
import numpy as np
import random
from gym import wrappers
from time import time

# 创建 Breakout 环境
env = gym.make("Breakout-v4", render_mode="rgb_array")

# 使用 Monitor 进行视频录制
#env = wrappers.RecordVideo(env, './videos/' + str(int(time())) + '/', name_prefix='DaZhuanKuai', episode_trigger=lambda episode_id: True)

# 初始化 Q 表
action_space_size = env.action_space.n
q_table = np.random.randn(84, 84, action_space_size)  # 假设状态离散化为一个简单的 84x84 网格

# 超参数
num_episodes = 50000
max_steps = 1000
learning_rate = 0.01
discount_factor = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.1

max_score = 0

# Q-learning 主循环
for episode in range(num_episodes):
    state, _ = env.reset()
    state = state[0]  # 处理 reset 返回值的元组
    state = state[0]  # 将状态处理为数组（仅在一些 gym 版本中需要）

    done = False
    total_reward = 0

    for step in range(max_steps):
        # 选择动作：ε-greedy 策略
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # 随机动作
        else:
            action = np.argmax(q_table[state[0], state[1]])  # 从 Q 表选择最大值动作

        # 执行动作并获取反馈
        next_state, reward, done, _, _ = env.step(action)

        # 更新 Q 表
        next_max = np.max(q_table[next_state[0], next_state[1]])  # 下一个状态的最大 Q 值
        q_table[state[0], state[1], action] = q_table[state[0], state[1], action] + learning_rate * (
            reward + discount_factor * next_max - q_table[state[0], state[1], action])

        state = next_state
        total_reward += reward

        if done:
            break

    # 减小探索率
    max_score = max(max_score, total_reward)
    
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    if episode % 500 == 499:
        print(f"Episode {episode+1}/{num_episodes}, Current Max Reward: {max_score}")

# 关闭环境并保存视频
env.close()
