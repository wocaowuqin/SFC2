#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
curriculum_scheduler.py
课程学习调度器

功能:
1. 根据难度排序请求
2. 渐进式增加训练难度
3. 自适应调整学习进度
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class CurriculumStage:
    """课程阶段定义"""
    name: str
    difficulty_range: Tuple[float, float]  # (min, max) difficulty
    num_episodes: int
    epsilon_range: Tuple[float, float]  # (start, end) epsilon
    expert_ratio: float


class DifficultyEstimator:
    """请求难度估计器"""

    def __init__(self, alpha=0.3, beta=0.3, gamma=0.4):
        self.alpha = alpha  # VNF数量权重
        self.beta = beta  # 带宽需求权重
        self.gamma = gamma  # 目标节点数权重

    def estimate(self, request: Dict) -> float:
        """
        估算请求难度

        公式:
        difficulty = α·norm(vnf_count) + β·norm(bandwidth) + γ·norm(dest_count)

        Returns:
            difficulty ∈ [0, 1]
        """
        vnf_count = len(request.get('vnf', []))
        bandwidth = request.get('bw_origin', 0)
        dest_count = len(request.get('dest', []))

        # 归一化 (假设最大值)
        vnf_norm = vnf_count / 8.0  # 最多8种VNF
        bw_norm = bandwidth / 10.0  # 最大带宽10
        dest_norm = dest_count / 5.0  # 最多5个目标

        difficulty = (
                self.alpha * vnf_norm +
                self.beta * bw_norm +
                self.gamma * dest_norm
        )

        return float(np.clip(difficulty, 0.0, 1.0))


class CurriculumScheduler:
    """
    课程学习调度器

    策略: 从简单到复杂的渐进式训练
    """

    def __init__(self, requests: List[Dict]):
        self.requests = requests
        self.estimator = DifficultyEstimator()

        # 为所有请求计算难度
        self.request_difficulties = [
            (req, self.estimator.estimate(req))
            for req in requests
        ]

        # 按难度排序
        self.request_difficulties.sort(key=lambda x: x[1])

        # 定义课程阶段
        self.stages = [
            CurriculumStage(
                name="Easy",
                difficulty_range=(0.0, 0.3),
                num_episodes=300,
                epsilon_range=(0.5, 0.3),
                expert_ratio=0.5
            ),
            CurriculumStage(
                name="Medium",
                difficulty_range=(0.3, 0.6),
                num_episodes=400,
                epsilon_range=(0.3, 0.15),
                expert_ratio=0.3
            ),
            CurriculumStage(
                name="Hard",
                difficulty_range=(0.6, 0.8),
                num_episodes=400,
                epsilon_range=(0.15, 0.05),
                expert_ratio=0.2
            ),
            CurriculumStage(
                name="Very Hard",
                difficulty_range=(0.8, 1.0),
                num_episodes=400,
                epsilon_range=(0.05, 0.01),
                expert_ratio=0.1
            )
        ]

        self.current_stage_idx = 0
        self.current_episode = 0

    def get_current_stage(self) -> CurriculumStage:
        """获取当前阶段"""
        return self.stages[self.current_stage_idx]

    def get_training_batch(self, batch_size: int = 100) -> List[Dict]:
        """
        获取当前阶段的训练批次

        Returns:
            batch: 符合当前难度范围的请求列表
        """
        stage = self.get_current_stage()
        min_diff, max_diff = stage.difficulty_range

        # 筛选符合难度范围的请求
        candidates = [
            req for req, diff in self.request_difficulties
            if min_diff <= diff <= max_diff
        ]

        # 随机采样
        if len(candidates) > batch_size:
            batch = np.random.choice(candidates, size=batch_size, replace=False).tolist()
        else:
            batch = candidates

        return batch

    def step(self):
        """推进课程进度"""
        self.current_episode += 1
        stage = self.get_current_stage()

        # 检查是否完成当前阶段
        if self.current_episode >= stage.num_episodes:
            if self.current_stage_idx < len(self.stages) - 1:
                self.current_stage_idx += 1
                self.current_episode = 0
                print(f"\n🎓 课程升级: 进入 {self.get_current_stage().name} 阶段")
            else:
                print(f"\n✅ 课程学习完成!")

    def get_epsilon(self) -> float:
        """获取当前epsilon (线性衰减)"""
        stage = self.get_current_stage()
        eps_start, eps_end = stage.epsilon_range

        # 线性插值
        progress = self.current_episode / max(1, stage.num_episodes)
        epsilon = eps_start + (eps_end - eps_start) * progress

        return float(epsilon)

    def get_expert_ratio(self) -> float:
        """获取当前专家混合比例"""
        return float(self.get_current_stage().expert_ratio)

    def print_status(self):
        """打印当前状态"""
        stage = self.get_current_stage()
        print(f"\n{'=' * 60}")
        print(f"课程状态")
        print(f"{'=' * 60}")
        print(f"阶段: {stage.name} ({self.current_stage_idx + 1}/{len(self.stages)})")
        print(f"难度范围: {stage.difficulty_range}")
        print(f"进度: {self.current_episode}/{stage.num_episodes}")
        print(f"Epsilon: {self.get_epsilon():.3f}")
        print(f"Expert Ratio: {self.get_expert_ratio():.2%}")
        print(f"{'=' * 60}\n")