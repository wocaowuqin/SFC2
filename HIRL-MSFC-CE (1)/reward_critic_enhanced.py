# reward_critic_enhanced.py
import logging
from typing import Dict, Optional, Tuple, Any
from collections import defaultdict

import numpy as np


logger = logging.getLogger(__name__)


class RewardCritic:
    """
    增强版 HRL 奖励系统 for 多播 VNF 映射（补丁版）
    - 参数化 self.params（便于实验调参）
    - 输入归一化与安全检查
    - DAgger 专家一致性增强
    - BackupPolicy 感知与自适应惩罚
    - Phase-aware 行为
    - 调试输出与统计接口
    """

    def __init__(self,
                 training_phase: int = 2,
                 epoch: int = 0,
                 max_epochs: int = 1200,
                 params: Optional[Dict[str, Any]] = None):
        # 基本训练阶段信息
        self.phase = int(training_phase)
        self.epoch = int(epoch)
        self.max_epochs = int(max_epochs)

        # 默认参数（可覆盖）
        default_params = {
            # high-level
            "high_goal_reward": 10.0,
            "high_goal_partial": 5.0,
            "high_cost_weight": 2.5,
            "high_blocking_bonus": 1.0,
            "high_blocking_penalty": 2.0,

            # low-level
            "low_subtask_reward": 3.0,
            "low_phase1_subtask": 5.0,
            "low_progress_weight": 1.5,
            # piecewise cost weights for ranges [0,0.3), [0.3,0.7), [0.7,1.0]
            "low_cost_weights": (0.5, 1.5, 3.0),
            "low_request_failed_phase1_penalty": 8.0,
            "low_failure_base": 4.0,

            # backup penalties (base)
            "backup_penalties": {
                "primary": 0.0,
                "resource_aware": 0.3,
                "smart_greedy": 0.8,
                "minimal": 1.5,
                "never_fail": 2.5,
                "unknown": 1.5
            },

            # qos weights
            "qos_weights": {"delay": 1.5, "bandwidth": 2.0, "jitter": 1.0, "packet_loss": 3.0},

            # reward clipping range
            "reward_clip": (-10.0, 10.0),

            # dagger tuning
            "dagger_consistency_scale": 1.2,
            "dagger_divergence_penalty": 0.3,

            # smoothing alpha for update_backup_stats
            "backup_alpha": 0.1
        }

        self.params = default_params if params is None else {**default_params, **params}

        # discount factors (kept for possible later use)
        self.gamma_high = 0.99
        self.gamma_low = 0.95

        # backup stats
        self.backup_success_rate = defaultdict(lambda: 0.5)  # EMA success rates
        self.backup_usage_count = defaultdict(int)

        # debug & logger
        self.debug = False
        self.logger = logger

        # quick refs
        self.reward_min, self.reward_max = self.params["reward_clip"]

    # -------------------------
    # Public helper / config
    # -------------------------
    def set_training_phase(self, phase: int, epoch: int = 0, max_epochs: Optional[int] = None):
        self.phase = int(phase)
        self.epoch = int(epoch)
        if max_epochs is not None:
            self.max_epochs = int(max_epochs)

    def set_params(self, **kwargs):
        """动态修改参数字典"""
        self.params.update(kwargs)
        # update reward_min/max if reward_clip changed
        if "reward_clip" in kwargs:
            self.reward_min, self.reward_max = self.params["reward_clip"]

    def set_debug(self, debug_on: bool):
        self.debug = bool(debug_on)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "epoch": self.epoch,
            "backup_success_rate": dict(self.backup_success_rate),
            "backup_usage_count": dict(self.backup_usage_count),
            "params_snapshot": {k: self.params[k] for k in ("reward_clip", "backup_penalties")}
        }

    # -------------------------
    # Input sanitation helpers
    # -------------------------
    def _clamp(self, x: Any, lo: float, hi: float) -> float:
        try:
            return float(np.clip(float(x), lo, hi))
        except Exception:
            return float(lo)

    # -------------------------
    # High-level reward
    # -------------------------
    def high_level_reward(self,
                          goal_reached: bool,
                          total_cost: float,
                          qos_satisfied: bool,
                          blocking_rate: float = 0.0) -> float:
        """
        Global-level reward for meta-controller.
        total_cost expected in [0,1] (normalized)
        blocking_rate expected in [0,1]
        """
        # sanitize
        total_cost = self._clamp(total_cost, 0.0, 1.0)
        blocking_rate = self._clamp(blocking_rate, 0.0, 1.0)

        p = self.params
        reward = 0.0

        if goal_reached and qos_satisfied:
            reward += p["high_goal_reward"]
            reward += 2.0 * (1.0 - total_cost)
        elif goal_reached:
            reward += p["high_goal_partial"]

        # global cost penalty
        reward -= p["high_cost_weight"] * total_cost

        # blocking health
        if blocking_rate < 0.1:
            reward += p["high_blocking_bonus"]
        elif blocking_rate > 0.5:
            reward -= p["high_blocking_penalty"]

        return self._normalize_reward(reward)

    # -------------------------
    # Low-level reward
    # -------------------------
    def low_level_reward(self,
                         sub_task_completed: bool,
                         step_cost: float,
                         request_failed: bool,
                         progress_to_goal: float,
                         backup_used: bool,
                         backup_level: str,
                         qos_violations: Optional[Dict[str, float]] = None,
                         failure_reason: Optional[str] = None) -> float:
        """
        Single-step reward for worker agent.
        - step_cost should be in [0,1]
        - progress_to_goal in [-1,1]
        - qos_violations values in [0,1] (ratios)
        - backup_level is string key into backup_penalties
        """
        # sanitize inputs
        step_cost = self._clamp(step_cost, 0.0, 1.0)
        progress_to_goal = self._clamp(progress_to_goal, -1.0, 1.0)
        backup_level = str(backup_level) if backup_level is not None else "unknown"
        if qos_violations is None:
            qos_violations = {}

        p = self.params
        reward = 0.0

        # Phase-aware base reward
        if self.phase == 1:
            if sub_task_completed:
                reward += p["low_phase1_subtask"]
            reward -= 0.3 * step_cost
            if request_failed:
                reward -= p["low_request_failed_phase1_penalty"]
        else:
            # completion reward + efficiency bonus
            if sub_task_completed:
                efficiency = max(0.0, 1.0 - step_cost)
                reward += p["low_subtask_reward"] + efficiency

            # progress shaping
            reward += p["low_progress_weight"] * progress_to_goal

            # piecewise cost penalty
            w1, w2, w3 = p["low_cost_weights"]
            if step_cost < 0.3:
                reward -= w1 * step_cost
            elif step_cost < 0.7:
                reward -= w2 * step_cost
            else:
                reward -= w3 * step_cost

            # dynamic failure penalty decays with epoch
            if request_failed:
                base_penalty = self._compute_failure_penalty(failure_reason)
                decay_factor = max(0.0, 1.0 - (self.epoch / max(1, self.max_epochs)))
                reward -= base_penalty * decay_factor

        # Backup penalty (adaptive)
        if backup_used:
            penalty = self._get_adaptive_backup_penalty(backup_level)
            reward -= penalty
        else:
            reward += 0.5  # main-path success bonus

        # QoS violations
        if qos_violations:
            qos_penalty = self._compute_qos_penalty(qos_violations)
            reward -= qos_penalty

        # final normalize / clip
        return self._normalize_reward(reward, clip_range=(-10.0, 6.0))

    # -------------------------
    # DAgger augmentation
    # -------------------------
    def dagger_augmented_reward(self,
                                base_reward: float,
                                agent_action: int,
                                expert_action: int,
                                state_novelty: float,
                                expert_confidence: float = 1.0) -> float:
        """
        Augment base_reward based on expert vs agent
        - state_novelty in [0,1], expert_confidence in [0,1]
        """
        # sanitize
        state_novelty = self._clamp(state_novelty, 0.0, 1.0)
        expert_confidence = self._clamp(expert_confidence, 0.0, 1.0)

        reward = float(base_reward)
        p = self.params

        if expert_confidence > 0.7:
            if agent_action == expert_action:
                consistency_bonus = p["dagger_consistency_scale"] * (1.0 - state_novelty) * expert_confidence
                reward += consistency_bonus
            else:
                reward -= p["dagger_divergence_penalty"] * expert_confidence
        else:
            if base_reward > 0 and state_novelty > 0.7:
                reward += 0.8  # encourage successful exploration in novel states

        return self._normalize_reward(reward, clip_range=(-10.0, 8.0))

    # -------------------------
    # helpers: failure & qos & backup
    # -------------------------
    def _compute_failure_penalty(self, reason: Optional[str]) -> float:
        penalty_map = {
            "resource_exhausted": 2.0,
            "timeout": 3.0,
            "routing_deadlock": 5.0,
            "qos_violation": 4.0,
            "invalid_action": 6.0
        }
        return float(penalty_map.get(reason, self.params.get("low_failure_base", 4.0)))

    def _compute_qos_penalty(self, violations: Dict[str, float]) -> float:
        penalty = 0.0
        weight_map = self.params.get("qos_weights", {})
        for metric, ratio in violations.items():
            try:
                r = float(ratio)
            except Exception:
                r = 0.0
            r = self._clamp(r, 0.0, 1.0)
            weight = float(weight_map.get(metric, 1.0))
            penalty += weight * r
        return float(penalty)

    def _get_adaptive_backup_penalty(self, backup_level: str) -> float:
        base_penalty_map = self.params.get("backup_penalties", {})
        base_pen = float(base_penalty_map.get(backup_level, base_penalty_map.get("unknown", 1.5)))
        success_rate = float(np.clip(self.backup_success_rate.get(backup_level, 0.5), 0.0, 1.0))
        adaptive_factor = 1.0 + (1.0 - success_rate)
        return base_pen * adaptive_factor

    def update_backup_stats(self, backup_level: str, success: bool):
        """
        EMA update of success rate. Called by environment after backup used.
        """
        alpha = float(self.params.get("backup_alpha", 0.1))
        prev = float(self.backup_success_rate.get(backup_level, 0.5))
        value = 1.0 if bool(success) else 0.0
        new = alpha * value + (1.0 - alpha) * prev
        self.backup_success_rate[backup_level] = float(new)
        self.backup_usage_count[backup_level] += 1

    # -------------------------
    # normalize / clip helpers
    # -------------------------
    def _normalize_reward(self, reward: float, clip_range: Optional[Tuple[float, float]] = None,
                          scale_to_unit: bool = False) -> float:
        if clip_range is None:
            clip_range = (self.reward_min, self.reward_max)
        lo, hi = float(clip_range[0]), float(clip_range[1])
        r = float(np.clip(float(reward), lo, hi))
        if scale_to_unit:
            if hi == lo:
                return r
            return 2.0 * (r - lo) / (hi - lo) - 1.0
        return r

    # -------------------------
    # main unified entry (compatible with previous interface)
    # -------------------------
    def criticize(self,
                  # low-level params
                  sub_task_completed: bool,
                  cost: float,
                  request_failed: bool,
                  progress_to_goal: float,
                  backup_used: bool,
                  backup_level: str,
                  qos_violations: Optional[Dict[str, float]] = None,
                  failure_reason: Optional[str] = None,
                  # DAgger params
                  agent_action: int = -1,
                  expert_action: int = -1,
                  state_novelty: float = 0.5,
                  expert_confidence: float = 1.0) -> float:
        """
        Unified critic entry. Returns final scalar reward.
        Keeps compatibility with previous call signatures.
        """
        # sanitize basic numeric inputs
        cost = self._clamp(cost, 0.0, 1.0)
        progress_to_goal = self._clamp(progress_to_goal, -1.0, 1.0)
        state_novelty = self._clamp(state_novelty, 0.0, 1.0)
        expert_confidence = self._clamp(expert_confidence, 0.0, 1.0)
        backup_level = str(backup_level) if backup_level is not None else "unknown"

        # compute base low-level reward
        base_reward = self.low_level_reward(
            sub_task_completed=sub_task_completed,
            step_cost=cost,
            request_failed=request_failed,
            progress_to_goal=progress_to_goal,
            backup_used=backup_used,
            backup_level=backup_level,
            qos_violations=qos_violations,
            failure_reason=failure_reason
        )

        # DAgger augmentation if actions provided
        if expert_action >= 0 and agent_action >= 0:
            final_reward = self.dagger_augmented_reward(
                base_reward=base_reward,
                agent_action=int(agent_action),
                expert_action=int(expert_action),
                state_novelty=state_novelty,
                expert_confidence=expert_confidence
            )
        else:
            final_reward = base_reward

        # debug logging of components
        if self.debug:
            try:
                self.logger.debug("RewardCritic.debug: " + str({
                    "phase": self.phase,
                    "epoch": self.epoch,
                    "sub_task_completed": sub_task_completed,
                    "cost": cost,
                    "progress": progress_to_goal,
                    "backup_used": backup_used,
                    "backup_level": backup_level,
                    "qos_violations": qos_violations,
                    "failure_reason": failure_reason,
                    "base_reward": base_reward,
                    "final_reward": final_reward,
                    "backup_success_rates": dict(self.backup_success_rate)
                }))
            except Exception:
                # best-effort logging, don't raise
                pass

        return float(final_reward)
