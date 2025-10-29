# -*- coding: utf-8 -*-
# @File    : config.py
# @Author  : ChatGPT-GPT5 (Completed by AI Assistant)
# @Desc    : Configuration for SFC-aware DRL Environment

import torch
import numpy as np
import time
from typing import List, Any


class Config:
    # ----------------------------------------------------------------------
    # 1. 训练和模型参数 (DRL HYPERPARAMETERS)
    # ----------------------------------------------------------------------
    LR = 1e-4  # 学习率 (Learning Rate)
    BATCH_SIZE = 64  # 批次大小
    MEMORY_CAPACITY = 10000  # 经验回放池容量
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DQN/DDQN 相关 (train_sfc.py 中引用)
    DISCOUNT = 0.95  # 折扣因子 (Gamma)
    N_STEPS = 1  # N-step Q-learning (set_nsteps 方法需要)
    TAU = 0.01  # 目标网络软更新率 (set_tau 方法需要)
    UPDATE_FREQUENCY = 100  # 目标网络更新频率 (set_update_frequency 方法需要)

    # Epsilon-greedy 策略 (train_sfc.py 中引用)
    E_GREEDY_EPISODES = 500  # Epsilon 衰减的回合数 (set_egreedy 方法需要)
    E_GREEDY_END = 0.01  # Epsilon 的最小值

    # ----------------------------------------------------------------------
    # 2. 环境和拓扑参数 (ENVIRONMENT & TOPOLOGY)
    # ----------------------------------------------------------------------
    TIME = time.strftime("%Y%m%d%H%M%S", time.localtime())  # 实验时间戳
    xml_topology_path = '/home/user1/PycharmProjects/SFC-master/mininet/topologies/topology2.xml'  # 拓扑文件路径 (需根据实际路径修改)
    pkl_weight_path = './traffic_pkl'  # 环境状态/流量变化的 pkl 文件目录

    # 环境/迭代次数 (train_sfc.py 中引用)
    EPISODES = 2000  # 训练的总回合数
    PKL_START = 0  # 从 pkl_weight_path 开始读取的索引
    PKL_CUT_NUM = 3000  # 读取 pkl 文件的总数 (定义一个 Epoch 的长度)
    PKL_STEP = 1  # 读取 pkl 文件的步长

    # SFC 服务链定义 (供 SFCEnv 使用，train_sfc.py 中通过 getattr 引用)
    # 假设一个包含 VNF CPU 需求的列表，SFCEnv 会基于此进行部署
    vnfs = [{'cpu': 10}, {'cpu': 5}, {'cpu': 7}]

    # 数据类型
    NUMPY_TYPE = np.float32
    A_STEP = 0.1
    B_STEP = 1.0

    # ----------------------------------------------------------------------
    # 3. 奖励权重和初始设定 (REWARD & CONSTRAINT WEIGHTS)
    # ----------------------------------------------------------------------
    BETA1 = 0.5  # 剩余带宽 (BW) 权重
    BETA2 = 0.3  # 时延 (Delay) 权重
    BETA3 = 0.2  # 丢包率 (Loss) 权重
    BETA4 = 0.4  # 节点资源剩余 (CPU) 权重
    BETA5 = 0.2  # 部署成本/能耗 (Cost) 权重

    # 奖励列表 (REWARD_DEFAULT)
    REWARD_DEFAULT: List[float] = [50, 1, -2, -10]

    # ----------------------------------------------------------------------
    # 4. 状态和动作空间 (在 Train.__init__ 中由拓扑设置)
    # ----------------------------------------------------------------------
    NUM_STATES = 7  # 状态通道数 (SFCEnv 中为 7)
    NUM_ACTIONS = 0  # 动作空间大小 (Train.__init__ 中设置为节点数)

    # ----------------------------------------------------------------------
    # 5. 动态设置方法 (SETTERS for Hyperparameter Sweep)
    # ----------------------------------------------------------------------
    @classmethod
    def set_num_states_actions(cls, num_states: int, num_actions: int):
        """设置状态和动作空间大小"""
        cls.NUM_STATES = num_states
        cls.NUM_ACTIONS = num_actions

    @classmethod
    def set_lr(cls, lr: float):
        cls.LR = lr

    @classmethod
    def set_nsteps(cls, nsteps: int):
        cls.N_STEPS = nsteps

    @classmethod
    def set_batchsize(cls, batchsize: int):
        cls.BATCH_SIZE = batchsize

    @classmethod
    def set_egreedy(cls, egreedy_episodes: int):
        cls.E_GREEDY_EPISODES = egreedy_episodes

    @classmethod
    def set_gamma(cls, gamma: float):
        cls.DISCOUNT = gamma

    @classmethod
    def set_update_frequency(cls, update_frequency: int):
        cls.UPDATE_FREQUENCY = update_frequency

    @classmethod
    def set_rewards(cls, rewards: List[float]):
        """设置奖励列表"""
        cls.REWARD_DEFAULT = rewards

    @classmethod
    def set_tau(cls, tau: float):
        cls.TAU = tau

    # ----------------------------------------------------------------------
    # 6. 日志记录 (LOGGING)
    # ----------------------------------------------------------------------
    @classmethod
    def log_params(cls, logger: Any):
        """记录当前配置参数到日志"""
        logger.info("==================== Config Parameters ====================")
        params_to_log = {
            "LR": cls.LR, "BATCH_SIZE": cls.BATCH_SIZE, "MEMORY_CAPACITY": cls.MEMORY_CAPACITY,
            "DEVICE": cls.DEVICE, "DISCOUNT (Gamma)": cls.DISCOUNT, "N_STEPS": cls.N_STEPS,
            "TAU": cls.TAU, "UPDATE_FREQUENCY": cls.UPDATE_FREQUENCY,
            "E_GREEDY_EPISODES": cls.E_GREEDY_EPISODES, "E_GREEDY_END": cls.E_GREEDY_END,
            "EPISODES": cls.EPISODES, "PKL_CUT_NUM": cls.PKL_CUT_NUM,
            "State Channels": cls.NUM_STATES, "Action Space (Nodes)": cls.NUM_ACTIONS,
            "Reward Weights (B1-B5)": [cls.BETA1, cls.BETA2, cls.BETA3, cls.BETA4, cls.BETA5],
            "Reward List": cls.REWARD_DEFAULT
        }
        for k, v in params_to_log.items():
            logger.info(f"{k}: {v}")
        logger.info("===========================================================")