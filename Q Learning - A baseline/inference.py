import gym
import numpy as np
import random

# 无限循环看测试结果
while True:
    # 加载训练好的 Q 表
    q_table = np.load("best_q_table.npy")
    epsilon = 0.1
    # 创建 Breakout 环境
    env = gym.make("Breakout-v4", render_mode="rgb_array", frameskip=15)

    # 推断阶段
    state, _ = env.reset()
    state = state[0]  # 处理 reset 返回值的元组
    state = state[0]  # 将状态处理为数组（仅在一些 gym 版本中需要）
    done = False
    tot_reward = 0
    while not done:
        # 选择动作：从 Q 表中选择最大值动作
        action = None
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # 随机动作
        else:
            action = np.argmax(q_table[state[0], state[1]])  # 从 Q 表选择最大值动作
        # 执行动作并获取反馈
        next_state, reward, done, _, _ = env.step(action)
        tot_reward += reward

        state = next_state

    env.close()
    if tot_reward  > 0:
        print(tot_reward)
