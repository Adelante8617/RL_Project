# 机器学习化学大作业：基于强化学习的Breakout-v4游戏模型评估

本仓库为 [*机器学习及其在化学中的应用*](https://dean.pku.edu.cn/service/web/courseDetail.php?flag=1&zxjhbh=BZ2425101035370_17962)的课程大作业。

# 文件结构

1. **Q Learning - A Baseline**
2.  **DQN**
3.  **Pretrained CNN For DQN**
4.  Making Charts
5.  requirements.txt

其中1-3是算法实现，4用于绘制数据可视化图表，5为项目依赖的包。

## 1. Q Learning - A Baseline

内部包含：
- demo.py , 一个基本的实现
- inference.py , 用于读取训练好的模型（保存在best_q_table.npy中）并展示训练结果
- best_q_table.npy , 保存的Q Table

## 2. DQN

内部包含：
- DQN.py , 定义了DQNModel，并将其封装为DQNAgent，用于完成训练等工作
- utils.py , 定义一些工具，包括预处理操作和记录训练日志
- train.py , 用于完成训练流程
- visualizing_video.py , 用于读取模型权重然后用于可视化
- checkpoints , 保存模型权重的文件夹

## 3. Pretrained CNN for DQN
名称与DQN文件夹相同的文件具有相同的功能

此外：
- DQN_using_resnet.py 为使用ResNet18作为预训练基座的模型实现
- ctrl.bat为循环执行train.py 的批处理文件，详细解释见报告。

## 4. Making Charts
用于生成统计图表和数据，运行making_chart.ipynb即可生成图像。

---

各python文件均可通过
```
python xx.py
```
方式运行。

若希望从头开始训练：
- 对于Baseline， 使用 `python demo.py`即可训练
- 对于另外两个， 使用 `python train.py`即可
- 对预训练DQN，也可通过文件夹内的批处理文件启动训练

若希望展示模型效果：
- 对于Baseline， 使用 `python inference.py`
- 对于另外两个， 使用 `python visuanlizing_video.py`即可

***默认的渲染模式是`render_mode='rgb_array'`，若希望观看模型实际游玩，将渲染模式修改为`'human'`即可。***

