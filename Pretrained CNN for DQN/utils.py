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

import torch
import torchvision.transforms as T

device = torch.device("cuda")

class PreprocessState:
    def __init__(self):
        # 预训练模型通常需要 RGB 图像和 224x224 尺寸
        self.transform = T.Compose([
            T.ToPILImage(),  # 将 numpy 数组或 Tensor 转换为 PIL 图像
            T.Grayscale(num_output_channels=3),  # 转为三通道灰度图
            T.Resize((224, 224)),  # 调整大小为 224x224
            T.ToTensor(),  # 转换为 Tensor
            # 标准化图像，假设使用的预训练模型（如 ResNet18）用的是 ImageNet 数据集的均值和标准差
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, state):
        """
        仅处理图像部分，忽略其他信息
        :param state: 状态，通常是一个 tuple，(图像, 其他信息)
        :return: 处理后的图像
        """
        image = state[0]  # 提取图像部分
        processed_image = self.transform(image).unsqueeze(0)  # 增加 batch 维度
        processed_image = processed_image
        return processed_image  # 返回处理后的图像



class TrainingLogger:
    def __init__(self, log_file, model, video_dir='videos', checkpoint_dir='checkpoints', log_interval=1, metric='score'):
        """
        初始化训练日志记录器
        :param log_file: 日志文件名，日志将写入此文件
        :param model: 模型，用于保存权重
        :param video_dir: 视频保存目录
        :param checkpoint_dir: 权重保存目录
        :param log_interval: 每训练多少轮记录一次日志
        :param metric: 记录的指标类型，'score' 表示记录最高得分，'loss' 表示记录最低loss
        """
        self.log_file = log_file
        self.model = model
        self.video_dir = video_dir
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.metric = metric
        self.start_time = None
        self.best_score = -float('inf')
        self.logger = self._setup_logger()
        self.version_counter = 0 # cnt for vis

        # 创建视频和模型存储的目录
        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _setup_logger(self):
        """
        设置日志记录器
        :return: 返回一个配置好的 logger 对象
        """
        logger = logging.getLogger('TrainingLogger')
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def start_training(self):
        """
        启动训练，记录开始时间
        """
        self.start_time = time.time()
        self.logger.info('Training started.')

    def log(self, epoch, metric_value, loss=None, env=None, episode=None, record_newest=False):
        """
        记录训练过程中的日志信息，并保存表现最好的模型和视频
        :param epoch: 当前训练轮次
        :param metric_value: 当前的得分或loss值
        :param loss: 当前的loss值（如果有的话）
        :param env: 游戏环境，用于保存视频
        :param episode: 当前回合，用于保存最好的视频
        :param record_newest: 用于保存当前回合的权重，默认为False
        """
        
        # 每log_interval轮记录一次
        if epoch % self.log_interval == 0:
            if self.metric == 'score':
                self.logger.info(f"Epoch {epoch}: {self.metric} = {metric_value}")
            elif self.metric == 'loss' and loss is not None:
                self.logger.info(f"Epoch {epoch}: loss = {loss}")
            else:
                self.logger.warning(f"Invalid metric or missing loss for epoch {epoch}.")

            # 如果当前得分超过历史最好的得分，保存视频和模型权重
            if metric_value > self.best_score:
                self.best_score = metric_value
                self.logger.info(f"New best score: {self.best_score}, saving model.")
                
                # 保存表现最好的模型权重
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, f'best_model_v{self.version_counter}.pth'))
                self.version_counter += 1
                # 保存表现最好的视频
            elif metric_value == self.best_score:
                self.logger.info(f"One more best score: {self.best_score}, saving model.")
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, f'best_model_v{self.version_counter}.pth'))
                
            if record_newest:
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, f'newest_model.pth'))
                

    def end_training(self):
        """
        结束训练，记录结束时间并输出训练总时长
        """
        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            self.logger.info(f'Training completed in {elapsed_time:.2f} seconds.')
        else:
            self.logger.warning('Training was not started correctly.')

    def save_best_video(self, env, episode):
        return ## 先不管
        """
        保存当前训练回合的视频
        :param env: 环境对象
        :param episode: 当前训练回合
        """
        frames = []
        state = env.reset()
        
        done = True
        while not done:
            frames.append(state)
            action = env.action_space.sample()  # 这里可以替换为训练策略
            state, _, done, _ = env.step(action)

        # 将帧保存为视频
        video_path = os.path.join(self.video_dir, f"best_episode_{episode}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        height, width, _ = frames[0].shape
        out = cv2.VideoWriter(video_path, fourcc, 30, (width, height))

        for frame in frames:
            out.write(frame)

        out.release()
        self.logger.info(f"Best episode video saved to: {video_path}")

            
if __name__ == "__main__":
    # Example usage:
    log_file = 'training_log.txt'
    logger = TrainingLogger(log_file, model=None, log_interval=5, metric='score')  # 每5轮记录一次，记录的是得分

    # 启动训练
    logger.start_training()

    # 模拟训练过程中的日志记录
    for epoch in range(1, 101):  # 假设训练有100轮
        # 模拟得分和loss的计算
        score = epoch * 0.1  # 这里是一个示例，你可以根据你的模型计算实际的得分
        loss = 100 - score  # 同样，这也是一个示例

        # 模拟环境和回合
        env = gym.make('Breakout-v4')
        episode = epoch

        # 记录每5轮的得分或loss
        logger.log(epoch, score, loss, env, episode)

    # 结束训练
    logger.end_training()
