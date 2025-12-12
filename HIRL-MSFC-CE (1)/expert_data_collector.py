#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
expert_data_collector.py
专家数据收集器

功能:
1. 使用环境内置的专家系统收集高质量轨迹
2. 自动过滤低质量数据
3. 保存为pickle格式供后续使用
"""
from typing import List, Dict, Optional  # 记得导入 Optional
import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
from collections import defaultdict

logger = logging.getLogger(__name__)


class ExpertDataCollector:
    """专家数据收集器"""

    def __init__(self, env, output_dir: Path):
        """
        Args:
            env: SFC_HIRL_Env_GNN 环境实例
            output_dir: 数据保存目录
        """
        self.env = env
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 数据统计
        self.stats = {
            'total_episodes': 0,
            'successful_episodes': 0,
            'total_transitions': 0,
            'high_confidence_transitions': 0,
            'low_confidence_transitions': 0,
            'failed_episodes': 0
        }

    def collect(self, num_episodes: int = 1000,
                min_confidence: float = 0.5,
                save_interval: int = 100) -> List[Dict]:
        """
        收集专家演示数据

        Args:
            num_episodes: 收集的episode数量
            min_confidence: 最小置信度阈值（过滤低质量数据）
            save_interval: 每隔多少episode保存一次

        Returns:
            expert_buffer: 专家轨迹列表
        """
        logger.info("=" * 60)
        logger.info("阶段1: 收集专家演示数据")
        logger.info("=" * 60)
        logger.info(f"目标episodes: {num_episodes}")
        logger.info(f"最小置信度: {min_confidence}")

        expert_buffer = []
        episode_count = 0

        while episode_count < num_episodes:
            # 收集一个episode的数据
            episode_data = self._collect_episode(min_confidence)

            if episode_data is not None:
                expert_buffer.extend(episode_data)
                self.stats['successful_episodes'] += 1
                self.stats['total_transitions'] += len(episode_data)

                logger.info(f"Episode {episode_count + 1}/{num_episodes}: "
                            f"收集了 {len(episode_data)} 个transitions")
            else:
                self.stats['failed_episodes'] += 1
                logger.debug(f"Episode {episode_count + 1}: 专家失败，跳过")

            episode_count += 1
            self.stats['total_episodes'] = episode_count

            # 定期保存
            if episode_count % save_interval == 0:
                self._save_buffer(expert_buffer, f"expert_data_ep{episode_count}.pkl")
                self._print_stats()

        # 最终保存
        self._save_buffer(expert_buffer, "expert_data_final.pkl")
        self._print_stats()

        logger.info(f"✅ 专家数据收集完成: {len(expert_buffer)} transitions")
        return expert_buffer

    def _collect_episode(self, min_confidence: float) -> Optional[List[Dict]]:
        """
        收集单个episode的专家数据

        Returns:
            episode_data: 该episode的所有transitions，如果失败则返回None
        """
        episode_data = []

        # 重置环境
        current_request, state = self.env.reset_request()
        if current_request is None:
            return None

        req_id = current_request.get('id', 'unknown')

        # Episode循环
        while len(self.env.unadded_dest_indices) > 0:
            # === 使用专家选择目标 ===
            flat_state = self.env._get_flat_state()
            expert_candidates = self.env.get_expert_high_level_candidates(
                flat_state, top_k=5
            )

            if not expert_candidates:
                # 专家无法处理，放弃这个episode
                logger.debug(f"Req {req_id}: 专家无法提供目标候选")
                return None

            # 选择专家最推荐的目标
            expert_goal, expert_goal_score = expert_candidates[0]

            # 置信度过滤
            if expert_goal_score < min_confidence:
                logger.debug(f"Req {req_id}: 专家置信度过低 ({expert_goal_score:.3f})")
                return None

            # === 获取有效动作 ===
            valid_actions = self.env.get_valid_low_level_actions()
            if not valid_actions:
                logger.debug(f"Req {req_id}: 没有有效动作")
                return None

            # === 记录当前状态（执行前） ===
            # 这里我们需要找出专家实际会选择的动作
            # 策略：让环境执行，然后从日志/返回值中提取专家动作

            # 保存状态用于后续构建transition
            current_state = state
            current_goal = expert_goal

            # === 尝试所有有效动作，找出专家实际选择的动作 ===
            # 这是一个技巧：我们通过环境的step来获知专家选择
            expert_action = self._infer_expert_action(
                current_goal, valid_actions
            )

            if expert_action is None or expert_action not in valid_actions:
                # 无法推断专家动作
                logger.debug(f"Req {req_id}: 无法推断专家动作")
                return None

            # === 执行专家动作 ===
            next_state, reward, sub_done, req_done = self.env.step_low_level(
                current_goal, expert_action
            )

            # === 构建transition ===
            transition = {
                'state': current_state,  # GNN状态 (x, ei, ea, req)
                'goal': current_goal,  # 目标节点索引
                'action': expert_action,  # 专家选择的动作
                'next_state': next_state,  # 下一个状态
                'reward': reward,  # 奖励
                'done': req_done,  # 是否结束
                'valid_actions': valid_actions,  # 有效动作列表
                'expert_confidence': expert_goal_score,  # 专家置信度
                'request_id': req_id  # 请求ID
            }

            # 标记置信度
            if expert_goal_score >= 0.8:
                self.stats['high_confidence_transitions'] += 1
            else:
                self.stats['low_confidence_transitions'] += 1

            episode_data.append(transition)

            # 更新状态
            state = next_state

            # 检查是否完成
            if req_done:
                # 检查是否成功（所有目标都完成）
                if len(self.env.unadded_dest_indices) == 0:
                    # 成功完成，返回数据
                    return episode_data
                else:
                    # 部分完成或失败
                    logger.debug(f"Req {req_id}: 未能完成所有目标")
                    return None

        # 正常完成
        return episode_data if len(self.env.unadded_dest_indices) == 0 else None

    def _infer_expert_action(self, goal: int, valid_actions: List[int]) -> Optional[int]:
        """
        推断专家会选择的动作（使用MSFCE_Solver专家系统）

        策略：
        1. 调用环境的专家方法获取最优动作
        2. 如果失败，使用启发式策略

        Args:
            goal: 目标节点
            valid_actions: 有效动作列表

        Returns:
            expert_action: 专家选择的动作索引，如果无法推断则返回None
        """
        # 方法1: 使用环境内置的专家方法
        try:
            # 环境已经有 expert_low_level_action 方法
            expert_action = self.env.expert_low_level_action(goal)
            if expert_action is not None and expert_action in valid_actions:
                return expert_action
        except Exception as e:
            logger.debug(f"Failed to get expert action: {e}")

        # 方法2: 使用专家系统的_calc_eval方法
        # 这是MSFCE_Solver的核心评估方法
        try:
            network_state = self.env._get_flat_state()
            best_action = None
            best_eval = -float('inf')

            # 评估每个有效动作
            for action in valid_actions[:5]:  # 限制评估数量，避免太慢
                # 解析动作到(p_idx, k_idx)
                # 根据你的动作空间定义来解析
                p_idx = action // self.env.K_path
                k_idx = action % self.env.K_path
                k = k_idx + 1  # 1-based

                try:
                    # 调用专家的评估函数
                    eval_val, paths, tree, hvt, feasible, _, _, _ = \
                        self.env.expert._calc_eval(
                            self.env.current_request,
                            goal,
                            k,
                            network_state
                        )

                    if feasible and eval_val > best_eval:
                        best_eval = eval_val
                        best_action = action
                except:
                    continue

            if best_action is not None:
                return best_action
        except Exception as e:
            logger.debug(f"Expert evaluation failed: {e}")

        # 方法3: 启发式 - 选择第一个有效动作
        if valid_actions:
            return valid_actions[0]

        return None

    def _save_buffer(self, buffer: List[Dict], filename: str):
        """保存数据到文件"""
        filepath = self.output_dir / filename

        with open(filepath, 'wb') as f:
            pickle.dump({
                'buffer': buffer,
                'stats': self.stats,
                'env_config': {
                    'n_nodes': self.env.n,
                    'n_links': self.env.L,
                    'n_vnf_types': self.env.K_vnf,
                    'n_actions': self.env.NB_LOW_LEVEL_ACTIONS,
                    'n_goals': self.env.NB_HIGH_LEVEL_GOALS
                }
            }, f)

        logger.info(f"💾 已保存数据到: {filepath}")

    def _print_stats(self):
        """打印统计信息"""
        logger.info("\n" + "=" * 60)
        logger.info("专家数据收集统计")
        logger.info("=" * 60)
        logger.info(f"总episodes:          {self.stats['total_episodes']}")
        logger.info(f"成功episodes:        {self.stats['successful_episodes']}")
        logger.info(f"失败episodes:        {self.stats['failed_episodes']}")
        logger.info(f"总transitions:       {self.stats['total_transitions']}")
        logger.info(f"高置信度:           {self.stats['high_confidence_transitions']}")
        logger.info(f"低置信度:           {self.stats['low_confidence_transitions']}")

        if self.stats['total_episodes'] > 0:
            success_rate = (self.stats['successful_episodes'] /
                            self.stats['total_episodes']) * 100
            logger.info(f"成功率:             {success_rate:.2f}%")

        logger.info("=" * 60 + "\n")

    @staticmethod
    def load_expert_data(filepath: Path) -> Tuple[List[Dict], Dict]:
        """
        加载专家数据

        Returns:
            buffer: 专家轨迹列表
            metadata: 元数据（统计信息、环境配置等）
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        logger.info(f"✅ 已加载专家数据: {len(data['buffer'])} transitions")
        return data['buffer'], data


# ============================================================================
# 使用示例
# ============================================================================
if __name__ == "__main__":
    import hyperparameters as H
    from hirl_sfc_env_gnn import SFC_HIRL_Env_GNN

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # 初始化环境
    env = SFC_HIRL_Env_GNN(
        input_dir=H.INPUT_DIR,
        topo=H.TOPOLOGY_MATRIX,
        dc_nodes=H.DC_NODES,
        capacities=H.CAPACITIES,
        use_gnn=True
    )

    # 创建收集器
    collector = ExpertDataCollector(env, output_dir=H.OUTPUT_DIR / "expert_data")

    # 收集数据
    expert_buffer = collector.collect(
        num_episodes=1000,
        min_confidence=0.5,
        save_interval=100
    )

    print(f"\n✅ 收集完成: {len(expert_buffer)} 个高质量transitions")