#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
expert_confidence_calculator.py (V6.1 Enterprise Edition - Fixed)

Enterprise-grade confidence scoring system for Expert Demonstrations.

Critical Fixes:
1. ✅ 修复 _compute_hybrid_confidence 参数类型错误
2. ✅ 修复 _record_decision_trace 可选参数问题
3. ✅ 添加缺失的导入和类型检查
4. ✅ 修复潜在的除零错误
5. ✅ 改进错误处理和日志

Usage:
    calc = ExpertConfidenceCalculator(config={'strategy': 'hybrid'})
    conf = calc.compute_confidence(cost, req_info, tree, episode_id="ep_1")
    calc.update_feedback(cost, conf, real_outcome=1.0)
"""

import numpy as np
import time
import math
import logging
import pickle
from collections import deque
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict

VERSION = "1.0.1"
AUTHOR = "ExpertConfidenceSystem"

logger = logging.getLogger(__name__)


# ============================================================
# 数据类定义
# ============================================================

@dataclass
class ConfidenceConfig:
    """置信度计算配置"""
    strategy: str = 'hybrid'

    # 基础参数
    min_conf: float = 0.25
    max_conf: float = 0.95
    base_scale: float = 50.0

    # 权重配置
    weight_base: float = 0.4
    weight_bayesian: float = 0.4
    weight_quality: float = 0.2

    # 调整因子
    complexity_weight: float = 0.05
    history_weight: float = 0.15

    # 噪声配置
    noise_std: float = 0.05
    adaptive_noise: bool = True

    # 校准配置
    enable_calibration: bool = True
    calibration_bins: int = 10
    min_samples_for_calibration: int = 20

    # 统计配置
    max_history: int = 1000
    performance_window: int = 100

    # 行为配置
    update_priors_in_inference: bool = False
    warmup_steps: int = 50


@dataclass
class DecisionRecord:
    """单个决策记录"""
    cost: float
    confidence: float
    timestamp: float
    episode_id: Optional[str] = None
    real_outcome: Optional[float] = None
    metadata: Dict = field(default_factory=dict)
    is_warmup: bool = False


# ============================================================
# 置信度校准器
# ============================================================

class ConfidenceCalibrator:
    """置信度校准器（确保置信度反映真实准确率）"""

    def __init__(self, n_bins: int = 10, min_samples: int = 20):
        self.n_bins = n_bins
        self.min_samples = min_samples
        self.bins = np.linspace(0, 1, n_bins + 1)
        self.stats = {i: {'sum_acc': 0.0, 'count': 0} for i in range(n_bins)}
        self.bias_history = deque(maxlen=100)
        self.correction_history = deque(maxlen=10)

    def update(self, confidence: float, outcome: float):
        """更新校准统计"""
        # ✅ 添加边界检查
        if not (0 <= confidence <= 1):
            logger.warning(f"Invalid confidence: {confidence}, clipping")
            confidence = np.clip(confidence, 0, 1)

        if not (0 <= outcome <= 1):
            logger.warning(f"Invalid outcome: {outcome}, clipping")
            outcome = np.clip(outcome, 0, 1)

        bias = confidence - outcome
        self.bias_history.append(bias)

        bin_idx = np.digitize(confidence, self.bins) - 1
        bin_idx = np.clip(bin_idx, 0, self.n_bins - 1)

        self.stats[bin_idx]['count'] += 1
        self.stats[bin_idx]['sum_acc'] += outcome

    def get_systematic_bias(self) -> float:
        """获取系统性偏差"""
        if not self.bias_history:
            return 0.0
        return float(np.mean(self.bias_history))

    def apply_calibration(self, confidence: float) -> float:
        """应用校准修正"""
        # 1. 分箱校准
        calibrated = self._apply_binning_calibration(confidence)

        # 2. 阻尼偏差修正
        sys_bias = self.get_systematic_bias()
        bias_mag = abs(sys_bias)
        total_samples = sum(s['count'] for s in self.stats.values())

        if bias_mag > 0.05 and total_samples > 100:
            # Sigmoid 强度控制
            bias_factor = 1.0 / (1.0 + math.exp(-10.0 * (bias_mag - 0.1)))
            correction_factor = 0.5 * bias_factor
            correction = correction_factor * sys_bias

            # 振荡阻尼
            self.correction_history.append(correction)
            if len(self.correction_history) >= 5:
                signs = [1 if c > 0 else -1 for c in self.correction_history]
                sign_changes = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])
                if sign_changes >= 3:
                    correction *= 0.5  # 如果振荡则减弱

            calibrated = calibrated - correction

        return np.clip(calibrated, 0.0, 1.0)

    def _apply_binning_calibration(self, confidence: float) -> float:
        """应用分箱校准"""
        bin_idx = np.digitize(confidence, self.bins) - 1
        bin_idx = np.clip(bin_idx, 0, self.n_bins - 1)

        stat = self.stats[bin_idx]
        count = stat['count']

        if count < self.min_samples:
            return confidence

        # ✅ 添加除零保护
        if count == 0:
            return confidence

        observed_acc = stat['sum_acc'] / count

        # Sigmoid 权重
        k = 1.0
        x = (count - self.min_samples) / float(max(1, self.min_samples))
        sigmoid_w = 1.0 / (1.0 + np.exp(-k * (x - 2.0)))
        alpha = min(0.8, sigmoid_w)

        return (1 - alpha) * confidence + alpha * observed_acc


# ============================================================
# 主类：专家置信度计算器
# ============================================================

class ExpertConfidenceCalculator:
    """
    企业级专家置信度计算器

    Features:
    - 多策略计算（基础、贝叶斯、混合）
    - 自适应调整和校准
    - 完整的状态管理和持久化
    - 性能监控和健康检查
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, custom_logger: Optional[logging.Logger] = None):
        """
        初始化置信度计算器

        Args:
            config: 配置字典
            custom_logger: 自定义日志器
        """
        self.logger = custom_logger or logging.getLogger(__name__)
        self._init_config(config)
        self._init_components()
        self._init_optimizations()

    def _init_config(self, config: Optional[Dict]):
        """初始化配置"""
        default_config = ConfidenceConfig()

        if config:
            for k, v in config.items():
                if hasattr(default_config, k):
                    self._validate_config_value(k, v)
                    setattr(default_config, k, v)
                else:
                    self.logger.warning(f"Unknown config key: {k}")

        # 交叉字段验证
        if default_config.min_conf >= default_config.max_conf:
            raise ValueError(f"min_conf ({default_config.min_conf}) must be < max_conf ({default_config.max_conf})")

        self.config = default_config
        self.logger.info(f"Config initialized: strategy={self.config.strategy}")

    def _validate_config_value(self, key: str, value: Any):
        """验证配置值"""
        if key in ['min_conf', 'max_conf']:
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{key} must be in [0, 1], got {value}")

        if key.startswith('weight_'):
            if value < 0:
                raise ValueError(f"{key} must be non-negative, got {value}")

        if key == 'base_scale':
            if value <= 0:
                raise ValueError(f"base_scale must be positive, got {value}")

    def _init_components(self):
        """初始化所有组件"""
        self.decision_history = deque(maxlen=self.config.max_history)
        self.performance_history = deque(maxlen=self.config.performance_window)

        # 贝叶斯状态
        self.bayesian_state = {
            'mean': 30.0,
            'precision': 1.0 / 2500.0
        }

        # 校准器
        if self.config.enable_calibration:
            self.calibrator = ConfidenceCalibrator(
                n_bins=self.config.calibration_bins,
                min_samples=self.config.min_samples_for_calibration
            )
        else:
            self.calibrator = None

        # 统计
        self.warmup_counter = 0
        self.perf_stats = {
            'total_calls': 0,
            'avg_time_ms': 0.0,
            'outliers_count': 0
        }

        # 测试模式
        self.test_mode = False
        self._original_noise_std = self.config.noise_std

    def _init_optimizations(self):
        """初始化优化组件"""
        self._stats_cache = {}  # 归一化统计缓存
        self._cache_hits = 0

    # ========== Public API: 生命周期管理 ==========

    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """
        热更新配置

        Args:
            new_config: 新的配置字典

        Returns:
            bool: 是否更新成功
        """
        try:
            # 预验证
            temp = ConfidenceConfig()
            for k, v in asdict(self.config).items():
                setattr(temp, k, v)

            for k, v in new_config.items():
                if hasattr(temp, k):
                    self._validate_config_value(k, v)
                    setattr(temp, k, v)

            if temp.min_conf >= temp.max_conf:
                raise ValueError("min_conf must be < max_conf")

            # 应用
            self.config = temp
            self.logger.info(f"Config updated: {new_config}")
            return True

        except Exception as e:
            self.logger.error(f"Config update failed: {e}")
            return False

    def set_test_mode(self, enabled: bool = True):
        """
        启用确定性测试模式

        Args:
            enabled: 是否启用
        """
        self.test_mode = enabled
        if enabled:
            np.random.seed(42)
            self.config.noise_std = 0.0
            self.logger.info("Test mode enabled (deterministic)")
        else:
            self.config.noise_std = self._original_noise_std
            self.logger.info("Test mode disabled")

    def get_internal_state(self) -> Dict[str, Any]:
        """获取内部状态（用于白盒测试）"""
        return {
            'bayesian_state': self.bayesian_state.copy(),
            'history_len': len(self.decision_history),
            'perf_history_avg': float(np.mean(self.performance_history)) if self.performance_history else 0.0,
            'cache_size': len(self._stats_cache),
            'warmup_counter': self.warmup_counter
        }

    def clear_cache(self):
        """清空缓存"""
        self._stats_cache.clear()
        self.logger.debug("Cache cleared")

    # ========== Public API: 推理 ==========

    def compute_confidence(self,
                           cost: float,
                           request_info: Optional[Dict] = None,
                           expert_tree: Optional[Dict] = None,
                           episode_id: Optional[str] = None) -> float:
        """
        计算置信度（主接口）

        Args:
            cost: 专家求解的 OTV 代价
            request_info: 请求信息字典
            expert_tree: 专家树结构
            episode_id: Episode ID

        Returns:
            float: 置信度 [min_conf, max_conf]
        """
        start_t = time.perf_counter()

        try:
            # 1. 预热阶段
            if self.warmup_counter < self.config.warmup_steps:
                self.warmup_counter += 1
                conf = self._compute_warmup_confidence(cost)
                self._record_decision_trace(cost, conf, request_info, episode_id, is_warmup=True)
                self._update_perf_stats(start_t)
                return conf

            # 2. 鲁棒性验证
            valid_cost, is_outlier = self._validate_cost_robust(cost)

            if is_outlier:
                self.perf_stats['outliers_count'] += 1
                self.logger.debug(f"Outlier detected: cost={cost:.2f}")
                raw_conf = 0.5  # 异常值默认中等置信度
            else:
                # 3. 策略计算
                if self.config.strategy == 'basic':
                    raw_conf = self._compute_basic_confidence(valid_cost, request_info, episode_id)
                elif self.config.strategy == 'bayesian':
                    raw_conf = self._compute_bayesian_confidence(valid_cost)
                elif self.config.strategy == 'hybrid':
                    raw_conf = self._compute_hybrid_confidence(valid_cost, request_info, expert_tree, episode_id)
                else:
                    self.logger.warning(f"Unknown strategy: {self.config.strategy}, using basic")
                    raw_conf = self._compute_basic_confidence(valid_cost, request_info, episode_id)

            # 4. 贝叶斯更新（推理阶段）
            if not is_outlier and self.config.update_priors_in_inference:
                self._update_bayesian_state(valid_cost)

            # 5. 校准
            if self.calibrator and self.config.enable_calibration:
                calibrated_conf = self.calibrator.apply_calibration(raw_conf)
            else:
                calibrated_conf = raw_conf

            # 6. 最终化
            final_conf = np.clip(calibrated_conf, self.config.min_conf, self.config.max_conf)
            self._record_decision_trace(valid_cost, final_conf, request_info, episode_id)

            self._update_perf_stats(start_t)
            return float(final_conf)

        except Exception as e:
            self.logger.error(f"Confidence computation failed: {e}", exc_info=True)
            # 降级策略：返回中等置信度
            return 0.5

    # ========== Public API: 学习 ==========

    def update_feedback(self, cost: float, confidence: float, real_outcome: float):
        """
        更新反馈（用于在线学习）

        Args:
            cost: 原始代价
            confidence: 之前预测的置信度
            real_outcome: 实际结果 (0-1)
        """
        try:
            valid_cost, is_outlier = self._validate_cost_robust(cost)

            # 更新校准器
            if self.calibrator:
                self.calibrator.update(confidence, real_outcome)

            # 更新性能历史
            self.performance_history.append(real_outcome)

            # 更新贝叶斯状态
            if not is_outlier:
                self._update_bayesian_state(valid_cost)

            # 清空缓存（历史已改变）
            if len(self._stats_cache) > 0:
                self.clear_cache()

        except Exception as e:
            self.logger.error(f"Feedback update failed: {e}", exc_info=True)

    # ========== Internal Logic: 策略实现 ==========

    def _compute_warmup_confidence(self, cost: float) -> float:
        """预热阶段的简单置信度"""
        if cost <= 0:
            return 0.9
        if cost >= 100:
            return 0.3
        return max(0.3, 0.9 - 0.006 * cost)

    def _validate_cost_robust(self, cost: float) -> Tuple[float, bool]:
        """
        双向异常值检测

        Returns:
            (处理后的cost, 是否异常值)
        """
        if not np.isfinite(cost):
            return 100.0, True

        is_outlier = False
        processed = cost

        if len(self.decision_history) > 50:
            costs = [r.cost for r in self.decision_history if np.isfinite(r.cost)]

            if len(costs) < 10:
                return cost, False

            q75, q25 = np.percentile(costs, [75, 25])
            iqr = q75 - q25

            # ✅ 添加除零保护
            if iqr < 1e-6:
                iqr = 1.0

            upper = max(q75 + 3.0 * iqr, 200.0)
            lower = min(q25 - 3.0 * iqr, -10.0)

            if cost > upper:
                is_outlier = True
                processed = min(cost, upper * 1.5)
            elif cost < lower:
                is_outlier = True
                processed = max(cost, lower * 0.5)
        elif cost < 0:
            # 早期阶段的负值检查
            is_outlier = True
            processed = 0.0

        return processed, is_outlier

    def _normalize_cost_context_aware(self, cost: float, episode_id: Optional[str]) -> float:
        """
        上下文感知的 Cost 归一化

        使用 MAD (Median Absolute Deviation) 进行鲁棒归一化
        """
        # 缓存检查
        cache_key = f"{episode_id}_{len(self.decision_history)}"

        if cache_key in self._stats_cache:
            median, mad = self._stats_cache[cache_key]
            self._cache_hits += 1
        else:
            # 计算统计
            if len(self.decision_history) < 10:
                return cost

            # 排除当前 episode 的数据（避免数据泄露）
            valid_costs = []
            for r in self.decision_history:
                if episode_id and r.episode_id == episode_id:
                    continue
                if np.isfinite(r.cost):
                    valid_costs.append(r.cost)

            if len(valid_costs) < 5:
                valid_costs = [r.cost for r in self.decision_history if np.isfinite(r.cost)]

            if len(valid_costs) < 3:
                return cost

            costs_arr = np.array(valid_costs)
            median = np.median(costs_arr)
            mad = np.median(np.abs(costs_arr - median))

            # ✅ MAD 保护
            if mad < 1e-3:
                mad = 1.0

            # 更新缓存（简单 LRU 清理）
            if len(self._stats_cache) > 20:
                self._stats_cache.clear()

            self._stats_cache[cache_key] = (median, mad)

        # Z-score 归一化（使用 MAD）
        z = (cost - median) / (1.4826 * mad)
        normalized = max(0.0, 20.0 + z * 20.0)

        return normalized

    def _compute_basic_confidence(self,
                                  cost: float,
                                  info: Optional[Dict],
                                  episode_id: Optional[str]) -> float:
        """基础策略：指数衰减 + 复杂度 + 历史"""
        # 归一化
        norm_cost = self._normalize_cost_context_aware(cost, episode_id)

        # 基础置信度
        base = 0.95 * np.exp(-norm_cost / self.config.base_scale)
        base = max(0.2, base)

        # 复杂度因子
        comp = 1.0
        if info and len(info.get('dest', [])) > 2:
            comp = 0.9

        # 历史因子
        hist = self._compute_history_factor_adaptive()

        # 综合
        conf = base * comp * hist

        # 噪声
        if self.config.adaptive_noise and not self.test_mode:
            noise = np.random.normal(0, self.config.noise_std)
            conf += noise

        return conf

    def _compute_bayesian_confidence(self, cost: float) -> float:
        """贝叶斯策略：基于后验分布"""
        mean = self.bayesian_state['mean']
        prec = self.bayesian_state['precision']
        obs_var = 100.0

        # 后验分布
        post_prec = prec + 1.0 / obs_var
        post_mean = (prec * mean + cost / obs_var) / post_prec
        post_std = np.sqrt(1.0 / post_prec)

        # ✅ 添加除零保护
        if post_std < 1e-6:
            post_std = 1.0

        # Z-score
        z = (mean - post_mean) / post_std

        # 数值稳定的 CDF
        if z < -8.0:
            prob = 0.0
        elif z > 8.0:
            prob = 1.0
        else:
            prob = 0.5 * (1 + math.erf(z / math.sqrt(2)))

        # S 曲线映射
        if prob <= 0.01:
            conf = 0.25
        elif prob >= 0.99:
            conf = 0.95
        else:
            x = 2.0 * prob - 1.0
            sigmoid = 1.0 / (1.0 + math.exp(-3.0 * x))
            conf = 0.25 + 0.7 * sigmoid

        return conf

    def _update_bayesian_state(self, cost: float):
        """更新贝叶斯先验"""
        obs_var = 100.0
        prior_mean = self.bayesian_state['mean']
        prior_prec = self.bayesian_state['precision']

        post_prec = prior_prec + 1.0 / obs_var
        post_mean = (prior_prec * prior_mean + cost / obs_var) / post_prec

        self.bayesian_state['mean'] = post_mean
        self.bayesian_state['precision'] = post_prec

    def _compute_hybrid_confidence(self,
                                   cost: float,
                                   info: Optional[Dict],  # ✅ 修复：添加类型注解
                                   tree: Optional[Dict],  # ✅ 修复：添加类型注解
                                   episode_id: Optional[str]) -> float:  # ✅ 修复：添加参数
        """混合策略：综合多种方法"""
        # 1. 基础置信度
        c_base = np.clip(self._compute_basic_confidence(cost, info, episode_id), 0, 1)

        # 2. 贝叶斯置信度
        c_bayes = np.clip(self._compute_bayesian_confidence(cost), 0, 1)

        # 3. 质量因子
        c_qual = 0.8
        if tree:
            depth = tree.get('depth', 0)
            if depth > 8:
                c_qual -= 0.1
            # 可以添加更多质量评估
        c_qual = np.clip(c_qual, 0, 1)

        # 4. 加权综合
        w_b = self.config.weight_base
        w_s = self.config.weight_bayesian
        w_q = self.config.weight_quality

        total = w_b + w_s + w_q

        # ✅ 添加除零保护
        if total < 1e-6:
            return (c_base + c_bayes + c_qual) / 3.0

        confidence = (w_b * c_base + w_s * c_bayes + w_q * c_qual) / total

        return confidence

    def _compute_history_factor_adaptive(self) -> float:
        """自适应历史性能因子"""
        n = len(self.performance_history)

        if n < 5:
            return 1.0

        # 指数衰减权重
        decay = 2.0

        # 如果最近波动大，增加衰减
        if n >= 20:
            recent_std = np.std(list(self.performance_history)[-20:])
            if recent_std > 0.3:
                decay = 3.0

        # 计算加权准确率
        weights = np.exp(np.linspace(-decay, 0, n))
        weights /= weights.sum()

        perf_array = np.array(self.performance_history)
        weighted_acc = np.sum(perf_array * weights)

        # 映射到因子
        w = self.config.history_weight
        factor = 1.0 + w * (2.0 * weighted_acc - 1.0)

        return np.clip(factor, 0.5, 1.5)

    def _record_decision_trace(self,
                               cost: float,
                               conf: float,
                               req: Optional[Dict],  # ✅ 修复：允许 None
                               episode_id: Optional[str],  # ✅ 修复：允许 None
                               is_warmup: bool = False):
        """记录决策轨迹"""
        rec = DecisionRecord(
            cost=cost,
            confidence=conf,
            timestamp=time.time(),
            metadata=req or {},  # ✅ 修复：处理 None
            episode_id=episode_id,
            is_warmup=is_warmup
        )
        self.decision_history.append(rec)

    def _update_perf_stats(self, start_t: float):
        """更新性能统计"""
        elapsed = (time.perf_counter() - start_t) * 1000  # ms

        n = self.perf_stats['total_calls']
        self.perf_stats['total_calls'] += 1

        # 增量均值更新
        if n == 0:
            self.perf_stats['avg_time_ms'] = elapsed
        else:
            self.perf_stats['avg_time_ms'] += (elapsed - self.perf_stats['avg_time_ms']) / (n + 1)

    # ========== Public API: 健康检查 ==========

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health = {
            'status': 'healthy',
            'warnings': []
        }

        # 检查贝叶斯状态
        if not np.isfinite(self.bayesian_state['mean']):
            health['status'] = 'degraded'
            health['warnings'].append('Bayesian state non-finite')

        if not np.isfinite(self.bayesian_state['precision']):
            health['status'] = 'degraded'
            health['warnings'].append('Bayesian precision non-finite')

        # 内存检查
        mem_est = len(self.decision_history) * 200  # bytes
        if mem_est > 10 * 1024 * 1024:  # 10MB
            health['warnings'].append(f'High memory: {mem_est / 1024 / 1024:.1f}MB')

        # 统计
        health['statistics'] = {
            'decisions': len(self.decision_history),
            'avg_latency_ms': self.perf_stats['avg_time_ms'],
            'cache_hits': self._cache_hits,
            'outliers_count': self.perf_stats['outliers_count']
        }

        return health

    def get_report(self) -> Dict[str, Any]:
        """生成详细报告"""
        if not self.decision_history:
            return {"error": "No decision history"}

        history = [r for r in self.decision_history if not r.is_warmup]

        if not history:
            return {"error": "No non-warmup decisions"}

        costs = [r.cost for r in history]
        confs = [r.confidence for r in history]

        report = {
            'total_decisions': len(history),
            'strategy': self.config.strategy,
            'cost_stats': {
                'min': float(np.min(costs)),
                'max': float(np.max(costs)),
                'mean': float(np.mean(costs)),
                'median': float(np.median(costs)),
                'std': float(np.std(costs)),
            },
            'confidence_stats': {
                'min': float(np.min(confs)),
                'max': float(np.max(confs)),
                'mean': float(np.mean(confs)),
                'median': float(np.median(confs)),
                'std': float(np.std(confs)),
            },
            'performance': {
                'avg_latency_ms': self.perf_stats['avg_time_ms'],
                'outliers': self.perf_stats['outliers_count'],
                'cache_hits': self._cache_hits
            }
        }

        # 校准误差
        if self.calibrator:
            report['calibration'] = {
                'systematic_bias': float(self.calibrator.get_systematic_bias()),
                'total_samples': sum(s['count'] for s in self.calibrator.stats.values())
            }

        return report

    def print_report(self):
        """打印人类可读报告"""
        report = self.get_report()

        if 'error' in report:
            print(f"Error: {report['error']}")
            return

        print("\n" + "=" * 70)
        print("EXPERT CONFIDENCE CALCULATOR REPORT (V6.1)")
        print("=" * 70)

        print(f"\nStrategy: {report['strategy']}")
        print(f"Total Decisions: {report['total_decisions']}")

        print(f"\nCost Statistics:")
        cs = report['cost_stats']
        print(f"  Range:  [{cs['min']:.2f}, {cs['max']:.2f}]")
        print(f"  Mean:   {cs['mean']:.2f}")
        print(f"  Median: {cs['median']:.2f}")

        print(f"\nConfidence Statistics:")
        cf = report['confidence_stats']
        print(f"  Range:  [{cf['min']:.3f}, {cf['max']:.3f}]")
        print(f"  Mean:   {cf['mean']:.3f}")
        print(f"  Median: {cf['median']:.3f}")

        print(f"\nPerformance:")
        pf = report['performance']
        print(f"  Avg Latency: {pf['avg_latency_ms']:.2f} ms")
        print(f"  Outliers:    {pf['outliers']}")
        print(f"  Cache Hits:  {pf['cache_hits']}")

        if 'calibration' in report:
            cal = report['calibration']
            print(f"\nCalibration:")
            print(f"  Systematic Bias: {cal['systematic_bias']:.4f}")
            print(f"  Total Samples:   {cal['total_samples']}")

        print("=" * 70 + "\n")

    # ========== 持久化 ==========

    def save_state(self, filepath: str):
        """保存状态到文件"""
        state = {
            'version': VERSION,
            'config': asdict(self.config),
            'bayesian_state': self.bayesian_state,
            'decision_history': list(self.decision_history),
            'performance_history': list(self.performance_history),
            'warmup_counter': self.warmup_counter,
            'perf_stats': self.perf_stats
        }

        if self.calibrator:
            state['calibrator'] = {
                'stats': self.calibrator.stats,
                'bias_history': list(self.calibrator.bias_history),
                'correction_history': list(self.calibrator.correction_history)
            }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        self.logger.info(f"State saved to {filepath}")

    @classmethod
    def load_state(cls, filepath: str) -> 'ExpertConfidenceCalculator':
        """从文件加载状态"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        # 创建实例
        calc = cls(config=state['config'])

        # 恢复状态
        calc.bayesian_state = state['bayesian_state']
        calc.decision_history = deque(state['decision_history'], maxlen=calc.config.max_history)
        calc.performance_history = deque(state['performance_history'], maxlen=calc.config.performance_window)
        calc.warmup_counter = state['warmup_counter']
        calc.perf_stats = state['perf_stats']

        # 恢复校准器
        if calc.calibrator and 'calibrator' in state:
            calc.calibrator.stats = state['calibrator']['stats']
            calc.calibrator.bias_history = deque(state['calibrator']['bias_history'], maxlen=100)
            calc.calibrator.correction_history = deque(state['calibrator']['correction_history'], maxlen=10)

        logger.info(f"State loaded from {filepath}")
        return calc


# ============================================================
# 便捷函数
# ============================================================

def create_confidence_calculator(strategy: str = 'hybrid', **kwargs) -> ExpertConfidenceCalculator:
    """
    便捷工厂函数

    Args:
        strategy: 'basic', 'bayesian', 'hybrid'
        **kwargs: 其他配置参数

    Returns:
        ExpertConfidenceCalculator 实例
    """
    config = {'strategy': strategy}
    config.update(kwargs)

    return ExpertConfidenceCalculator(config=config)


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("ExpertConfidenceCalculator V6.1 - Test")
    print("=" * 70)

    # 创建计算器
    calc = create_confidence_calculator(
        strategy='hybrid',
        min_conf=0.25,
        max_conf=0.95
    )

    # 模拟决策
    np.random.seed(42)

    print("\nSimulating 100 decisions...")
    for i in range(100):
        cost = np.random.exponential(40)
        request_info = {
            'dest': list(range(np.random.randint(1, 6))),
            'vnf': list(range(np.random.randint(2, 5))),
            'bw_origin': np.random.uniform(1, 10)
        }

        conf = calc.compute_confidence(cost, request_info, episode_id=f"ep_{i}")

        # 模拟反馈
        outcome = 1.0 if cost < 40 else 0.5 if cost < 70 else 0.0
        calc.update_feedback(cost, conf, outcome)

        if i % 20 == 0:
            print(f"  Step {i:3d}: cost={cost:6.2f}, confidence={conf:.3f}")

    # 报告
    print("\n")
    calc.print_report()

    # 健康检查
    health = calc.health_check()
    print(f"Health Status: {health['status']}")
    if health['warnings']:
        print(f"Warnings: {health['warnings']}")

    print("\n✅ Test completed successfully!")