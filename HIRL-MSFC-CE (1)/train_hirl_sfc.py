#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File    : train_hirl_sfc.py
# ---------- Defensive patch: ensure IPython has expected attrs ----------
# Put this BEFORE any `import matplotlib` / `import matplotlib.pyplot` / from matplotlib.figure import Figure
try:
    import importlib

    # only try to import IPython if present; do NOT force-install
    spec = importlib.util.find_spec("IPython")
    if spec is not None:
        import IPython

        # If IPython is present but missing attributes used by matplotlib, add safe defaults
        if not hasattr(IPython, "version_info"):
            # Matplotlib only needs to compare a tuple slice like version_info[:2],
            # provide a minimal tuple so comparisons won't error.
            IPython.version_info = (0, 0, 0)
        if not hasattr(IPython, "get_ipython"):
            IPython.get_ipython = lambda: None
except Exception:
    # Any errors here are non-fatal; continue without IPython
    pass
# ---------------------------------------------------------------------
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号乱码
import numpy as np
from collections import namedtuple
import os

# ---
# 步骤 1: 在文件顶部添加导入
# ---
import matplotlib

matplotlib.use('Agg')  # 关键！避免所有 IPython 相关错误
import pandas as pd

# 修复: (补丁 9) 导入新的 Matplotlib 模块
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# (补丁 4, 6, 8, 9) 新增导入
import logging
import traceback
import sys  # 修复: (补丁 3/9) 导入 sys

# ----------------------------------------------------
# 修复: (补丁 3/9) 设置 UTF-8 输出
# ----------------------------------------------------
try:
    # Python 3.7+ 支持 reconfigure
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    # 在某些嵌套环境中可能不支持 reconfigure，忽略即可
    pass

# 导入 Keras/TF 和辅助工具
from tensorflow.keras.utils import to_categorical

# 导入所有自定义模块
import hyperparameters as H
from hirl_sfc_env import SFC_HIRL_Env
from hirl_sfc_models import MetaControllerNN, Hdqn_SFC
from hirl_sfc_agent import Agent_SFC

# (来自 atari)
ActorExperience = namedtuple("ActorExperience", ["state", "goal", "action", "reward", "next_state", "done"])


# ============================================
# (补丁 8) 配置验证
# ============================================
def validate_configuration():
    """验证配置参数的合理性"""
    errors = []
    warnings = []

    # 检查路径
    if not H.INPUT_DIR.exists():
        errors.append(f"输入目录不存在: {H.INPUT_DIR}")

    # 检查超参数
    if H.BATCH_SIZE <= 0:
        errors.append(f"批大小必须为正数: {H.BATCH_SIZE}")

    if H.GAMMA < 0 or H.GAMMA > 1:
        errors.append(f"折扣因子必须在 [0, 1]: {H.GAMMA}")

    if H.LR <= 0:
        warnings.append(f"学习率很小: {H.LR}")

    if H.PRE_TRAIN_STEPS > H.STEPS_LIMIT:
        warnings.append("预训练步数大于总步数限制")

    # 打印结果
    if errors:
        print("配置错误:")
        for e in errors:
            print(f"  - {e}")
        return False

    if warnings:
        print("配置警告:")
        for w in warnings:
            print(f"  - {w}")

    print("配置验证通过")
    return True


# ============================================
# (补丁 6, 9) 改进的绘图和保存 (采纳用户的建议)
# ============================================

# (日志记录器将在 main_improved 中初始化)
logger = logging.getLogger(__name__)


def save_training_plots(metrics_csv_path: str, out_dir: str):
    """
    (最终修复版) 读取 CSV 并使用面向对象的 Matplotlib API 保存图表。
    已添加针对 'IPython.version_info' 错误的临时补丁。
    """
    # --- 补丁开始: 临时隐藏 IPython 以绕过 matplotlib 的版本检查错误 ---
    import sys
    original_ipython = sys.modules.get('IPython')
    if 'IPython' in sys.modules:
        del sys.modules['IPython']
    # ------------------------------------------------------------

    try:
        if not os.path.exists(metrics_csv_path):
            logger.warning("训练指标 CSV 未找到，跳过绘图: %s", metrics_csv_path)
            return False

        df = pd.read_csv(metrics_csv_path)
        if df.empty:
            logger.warning("训练指标 CSV 为空，跳过绘图: %s", metrics_csv_path)
            return False

        os.makedirs(out_dir, exist_ok=True)

        # 平滑函数
        def smooth(values, window=100):
            if len(values) == 0: return []
            if len(values) < window:
                window = max(1, len(values) // 10)
                if window == 0: window = 1
            return pd.Series(values).rolling(window, min_periods=1).mean()

        def do_plot(col, title, ylabel, use_smooth=False):
            if col not in df.columns:
                logger.debug("CSV 不包含列: %s", col)
                return

            # 1. 创建 Figure (画布) 和 Axes (坐标轴)
            fig = Figure(figsize=(10, 5))
            ax = fig.add_subplot(1, 1, 1)

            # 2. 在 Axes 上绘图
            if use_smooth:
                ax.plot(df[col].values, alpha=0.3, label='原始')
                # 确保平滑窗口不超过数据长度
                ax.plot(smooth(df[col].values), label='平滑 (窗口=100)', linewidth=2)
                ax.legend()
            else:
                ax.plot(df[col].values)

            # 3. 设置属性
            ax.set_title(title)
            ax.set_xlabel("记录索引 (回合)")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            # 4. 保存文件
            fname = os.path.join(out_dir, f"{col}_trend.png")
            canvas = FigureCanvas(fig)
            canvas.print_figure(fname, dpi=150)
            logger.info("已保存图表: %s", fname)

        # --- 绘图 ---
        do_plot('reward', title='回合奖励', ylabel='Reward', use_smooth=True)
        do_plot('acceptance_rate', title='请求接受率', ylabel='Acceptance Rate (%)')
        # ✅ 新增: 阻塞率绘图
        do_plot('blocking_rate', title='业务请求阻塞率', ylabel='Blocking Rate (%)')

        do_plot('avg_cpu_util', title='CPU 利用率', ylabel='Avg Util (%)', use_smooth=True)
        do_plot('avg_mem_util', title='内存 利用率', ylabel='Avg Util (%)', use_smooth=True)
        do_plot('avg_bw_util', title='带宽 利用率', ylabel='Avg Util (%)', use_smooth=True)

        logger.info("所有图表已生成")
        return True

    except Exception as e:
        logger.exception("生成图表失败: %s", e)
        return False

    finally:
        # --- 补丁结束: 恢复 IPython ---
        if original_ipython:
            sys.modules['IPython'] = original_ipython


def save_training_data_safely(tracking_data, output_dir):
    """安全保存训练数据并调用绘图"""
    csv_path_str = ""
    try:
        # 1. 检查数据是否为空
        if not tracking_data or all(len(v) == 0 for v in tracking_data.values()):
            logger.warning("没有收集到训练数据, 无法保存 CSV 或绘图")
            return False

        # 2. 保存 CSV
        df = pd.DataFrame(tracking_data)
        csv_path = output_dir / "training_metrics.csv"
        csv_path_str = str(csv_path)
        df.to_csv(csv_path, index=False)
        logger.info(f"训练指标已保存到 {csv_path}")

        # 3. 修复: 在保存 CSV *之后* 调用新的绘图函数
        save_training_plots(csv_path_str, str(output_dir))

        return True

    except Exception as e:
        logger.error(f"保存数据失败: {e}")
        traceback.print_exc()
        return False


# ============================================
# (补丁 4, 8) 改进的主函数
# ============================================
def main_improved():
    """改进的主训练函数"""

    # 0. 验证配置
    if not validate_configuration():
        print("配置验证失败,退出")
        return

    # 1. 设置日志
    if not H.OUTPUT_DIR.exists():
        H.OUTPUT_DIR.mkdir(parents=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(H.OUTPUT_DIR / 'training.log', encoding='utf-8'),  # 修复: 文件也指定 utf-8
            logging.StreamHandler()
        ]
    )
    # (现在全局日志记录器已配置)
    global logger
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("开始 HIRL-SFC 训练")
    logger.info("=" * 60)

    # (定义 tracking_data 和 hdqn_net 在 try 块之外，以便 finally 可以访问)
    tracking_data = {
        'episode': [], 'reward': [],
        'acceptance_rate': [],
        'blocking_rate': [],  # ✅ 新增: 记录阻塞率
        'avg_cpu_util': [], 'avg_mem_util': [], 'avg_bw_util': []
    }
    hdqn_net = None

    try:
        # --- 3. 初始化 ---
        logger.info("--- 2. 初始化环境、专家和代理 ---")
        env = SFC_HIRL_Env(H.INPUT_DIR, H.TOPOLOGY_MATRIX, H.DC_NODES, H.CAPACITIES)

        # 获取SFC环境的特定维度
        STATE_SHAPE = env.observation_space.shape
        NB_GOALS = env.NB_HIGH_LEVEL_GOALS
        NB_ACTIONS = env.NB_LOW_LEVEL_ACTIONS

        # 高层
        metacontroller = MetaControllerNN(state_shape=STATE_SHAPE, n_goals=NB_GOALS, lr=H.LR)

        # 低层 (H-DQN 架构)
        hdqn_net = Hdqn_SFC(state_shape=STATE_SHAPE, n_goals=NB_GOALS, n_actions=NB_ACTIONS, lr=H.LR)
        low_level_agent = Agent_SFC(
            net=hdqn_net,
            n_actions=NB_ACTIONS,
            mem_cap=H.EXP_MEMORY,
            exploration_steps=H.EXPLORATION_STEPS,
            train_freq=H.TRAIN_FREQ,
            hard_update=H.HARD_UPDATE_FREQUENCY,
            n_samples=H.BATCH_SIZE,
            gamma=H.GAMMA
        )
        low_level_agent.compile()  # 构建 Keras 训练模型

        logger.info(f"状态向量大小: {STATE_SHAPE}")
        logger.info(f"高层目标(子任务)数量: {NB_GOALS}")
        logger.info(f"低层动作数量: {NB_ACTIONS}")

        # --- 4. 阶段 1: 模仿学习 (预训练) ---
        logger.info(f"--- 3. 阶段 1: 模仿学习预训练 ( {H.PRE_TRAIN_STEPS} 步) ---")
        # (我们假设 PRE_TRAIN_STEPS 已被设置为一个合理的值, e.g., 200)
        stepCount = 0
        current_request, high_level_state = env.reset_request()

        t = 0
        failed_requests = 0  # (补丁 4)

        while t < H.PRE_TRAIN_STEPS:
            try:  # (补丁 4)
                if current_request is None:
                    current_request, high_level_state = env.reset_request()
                    if current_request is None:
                        logger.warning("没有更多请求,提前结束预训练")  # (Emoji removed)
                        break
                    continue

                # 获取专家轨迹
                try:
                    _, expert_traj = env.expert.solve_request_for_expert(
                        current_request, env._get_network_state_dict()
                    )
                except Exception as e:
                    logger.error(f"专家求解失败: {e}")
                    failed_requests += 1
                    current_request, high_level_state = env.reset_request()
                    continue

                if not expert_traj:
                    failed_requests += 1
                    if failed_requests % 100 == 0:
                        logger.warning(f"警告: 已有 {failed_requests} 个请求失败")  # (Emoji removed)
                    current_request, high_level_state = env.reset_request()
                    continue

                failed_requests = 0  # 重置失败计数

                # 遍历专家轨迹的所有步骤
                for step_idx, (exp_goal, exp_action_tuple, exp_cost) in enumerate(expert_traj):
                    if exp_goal not in env.unadded_dest_indices:
                        continue

                    exp_action = exp_action_tuple[0] * env.K_path + exp_action_tuple[1]

                    # 在环境中执行
                    next_high_level_state, cost, sub_task_done, request_done = env.step_low_level(exp_goal, exp_action)

                    # 存储
                    reward = low_level_agent.criticize(sub_task_completed=True, cost=cost, request_failed=False)
                    goal_one_hot = to_categorical(exp_goal, num_classes=NB_GOALS)
                    exp = ActorExperience(high_level_state, goal_one_hot, exp_action, reward, next_high_level_state,
                                          sub_task_done)
                    low_level_agent.store(exp)

                    # 训练
                    if t % low_level_agent.trainFreq == 0:
                        low_level_agent.update(t)

                    metacontroller.collect(high_level_state, exp_goal)
                    if t % H.META_TRAIN_FREQ == 0 and metacontroller.check_training_clock():
                        metacontroller.train()

                    high_level_state = next_high_level_state
                    t += 1

                    if t % 1000 == 0:
                        logger.info(f"预训练... {t}/{H.PRE_TRAIN_STEPS}")

                    if t >= H.PRE_TRAIN_STEPS:  # 确保我们不会超出步数限制
                        break

                    if request_done:
                        break

                # 预训练循环后重置请求
                current_request, high_level_state = env.reset_request()

            except Exception as e:
                logger.error(f"预训练循环出错: {e}")
                traceback.print_exc()
                # 尝试恢复
                current_request, high_level_state = env.reset_request()
                continue

        logger.info(f"预训练完成: {t} 步, {failed_requests} 个失败请求")

        # --- 4. 阶段 2: 混合 IL/RL 训练 ---
        logger.info("--- 4. 阶段 2: 混合 IL/RL 训练 ---")
        low_level_agent.randomPlay = False
        stepCount = t  # 从预训练的步数继续
        episodeCount = 0
        # total_requests_arrived = 0 (已在 env 中)
        # total_requests_served = 0 (已在 env 中)

        while episodeCount < H.EPISODE_LIMIT and stepCount < H.STEPS_LIMIT and env.t < env.T:

            try:  # (补丁 4) 捕获回合级错误
                current_request, high_level_state = env.reset_request()
                if current_request is None:
                    logger.warning("仿真结束, 提前停止训练")  # (Emoji removed)
                    break

                # total_requests_arrived += 1 (已移至 env.reset_request)
                request_done = False
                episode_reward = 0
                episode_steps = 0

                while not request_done:
                    # --- A. 高层决策 (元控制器) ---
                    high_level_state_v = np.reshape(high_level_state, (1, -1))
                    goal_probs = metacontroller.predict(high_level_state_v)  # < 已修复 (补丁 2)
                    goal = metacontroller.sample(goal_probs)

                    true_goal = env.get_expert_high_level_goal(high_level_state_v)
                    metacontroller.collect(high_level_state, true_goal)  # (DAgger: 总是收集专家答案)

                    # ----------------------------------------------------
                    # 修复: 强制使用专家的 'true_goal'
                    # 这将绕过高层代理的 "DAgger 检查"，并强制开始训练低层代理
                    # (原始代码: if goal != true_goal: ... continue)
                    # ----------------------------------------------------
                    goal = true_goal

                    if goal not in env.unadded_dest_indices:
                        # (如果专家给出的目标也无效，则跳过)
                        logger.warning(f"专家给出的目标 {goal} 无效或已完成, 丢弃轨迹")
                        request_done = True
                        continue

                    goal_one_hot = np.reshape(to_categorical(goal, num_classes=NB_GOALS), (1, -1))

                    # --- B. 低层执行 (代理) ---
                    sub_task_done = False
                    low_level_state = high_level_state
                    low_level_timeout_counter = 0  # 修复: 添加超时计数器

                    while not sub_task_done:

                        # 修复: 检查无限循环 (如果所有 K 路径都不可行)
                        low_level_timeout_counter += 1
                        if low_level_timeout_counter > (env.K_path + 3):  # 尝试 K_path 次 + 缓冲区
                            logger.warning(f"低层超时: 无法为 goal {goal} 找到可行路径。")
                            request_done = True  # 放弃整个请求
                            break

                        low_level_state_v = np.reshape(low_level_state, (1, -1))

                        valid_actions = env.get_valid_low_level_actions()
                        if not valid_actions:
                            action = 0  # 兜底
                        else:
                            action = low_level_agent.selectMove(low_level_state_v, goal_one_hot, valid_actions)

                        # 修复: 调用调试工具
                        validate_action_space(env, action)

                        # B2. 环境执行
                        next_low_level_state, cost, sub_task_done, request_done = env.step_low_level(goal, action)

                        # B3. 计算内部奖励 (RL)
                        reward = low_level_agent.criticize(sub_task_done, cost, request_failed=(not sub_task_done))
                        episode_reward += reward

                        # B4. 存储经验
                        exp = ActorExperience(high_level_state, goal_one_hot.flatten(), action, reward,
                                              next_low_level_state,
                                              sub_task_done)
                        low_level_agent.store(exp)

                        # B5. 训练
                        if stepCount % low_level_agent.trainFreq == 0:
                            loss, avgQ, avgTD = low_level_agent.update(stepCount)  # < 已修复 (补丁 1)
                            if stepCount % 1000 == 0 and (avgQ != 0 or loss != 0):
                                logger.info(f"Step {stepCount} | Q: {avgQ:.3f}, TD: {avgTD:.3f}, Loss: {loss:.3f}")

                        if stepCount % H.META_TRAIN_FREQ == 0 and metacontroller.check_training_clock():
                            metacontroller.train()

                        low_level_agent.annealControllerEpsilon(stepCount)

                        low_level_state = next_low_level_state
                        stepCount += 1
                        episode_steps += 1

                        if request_done:
                            break

                    high_level_state = low_level_state

                # --- 步骤 3: 在回合结束时 (print 之前) 收集数据 ---
                # (total_requests_served 已在 env.step_low_level 中更新)

                avg_cpu_util = (1.0 - np.mean(env.C) / env.C_cap) * 100.0
                avg_mem_util = (1.0 - np.mean(env.M) / env.M_cap) * 100.0
                avg_bw_util = (1.0 - np.mean(env.B) / env.B_cap) * 100.0

                tracking_data['episode'].append(episodeCount)
                tracking_data['reward'].append(episode_reward)

                # --- 计算并记录阻塞率和接受率 ---
                total_reqs = max(1, env.total_requests_seen)
                req_accept_rate = (env.total_requests_accepted / total_reqs) * 100.0

                # 阻塞请求数 sr = 总请求数 s - 成功部署数
                blocked_requests = env.total_requests_seen - env.total_requests_accepted
                blocking_rate = (blocked_requests / total_reqs) * 100.0

                dest_accept_rate = (env.total_dest_accepted / max(1, env.total_dest_seen)) * 100.0

                tracking_data['acceptance_rate'].append(req_accept_rate)
                tracking_data['blocking_rate'].append(blocking_rate)  # ✅ 记录阻塞率

                tracking_data['avg_cpu_util'].append(avg_cpu_util)
                tracking_data['avg_mem_util'].append(avg_mem_util)
                tracking_data['avg_bw_util'].append(avg_bw_util)

                logger.info(f"--- 回合 {episodeCount} (T={env.t}) ---")
                logger.info(f"总步数: {stepCount}, Epsilon: {low_level_agent.controllerEpsilon:.4f}")
                logger.info(f"回合奖励: {episode_reward:.3f}, 回合步数: {episode_steps}")

                logger.info(
                    f"当前接受率(完整请求): {req_accept_rate:.2f}% ({env.total_requests_accepted}/{env.total_requests_seen})")
                # ✅ 打印阻塞率
                logger.info(
                    f"当前阻塞率: {blocking_rate:.2f}% ({blocked_requests}/{env.total_requests_seen})")

                logger.info(
                    f"当前接受率(按目的地): {dest_accept_rate:.2f}% ({env.total_dest_accepted}/{env.total_dest_seen})")
                logger.info(
                    f"当前资源利用率 CPU: {avg_cpu_util:.2f}%, MEM: {avg_mem_util:.2f}%, BW: {avg_bw_util:.2f}%")

                episodeCount += 1

                if episodeCount % 50 == 0:
                    logger.info("保存模型...")
                    model_path = H.OUTPUT_DIR / f"sfc_hirl_model_ep{episodeCount}"
                    hdqn_net.saveWeight(str(model_path))

                if episodeCount >= H.EPISODE_LIMIT:
                    logger.info(f"达到回合数限制 {H.EPISODE_LIMIT}, 停止训练。")
                    break

            except Exception as e:
                logger.error(f"主训练回合 {episodeCount} 失败: {e}")
                traceback.print_exc()
                # 尝试恢复，进入下一个回合
                episodeCount += 1
                continue

        logger.info("--- 5. 训练完成 ---")

    except KeyboardInterrupt:
        logger.warning("用户中断训练")  # (Emoji removed)
        # 保存当前进度
        if hdqn_net is not None:
            hdqn_net.saveWeight(str(H.OUTPUT_DIR / "interrupted_model"))
            logger.info("已保存中断的模型。")

    except Exception as e:
        logger.error(f"训练期间发生致命错误: {e}")
        logger.error(traceback.format_exc())
        raise

    finally:
        logger.info("--- 6. 生成分析图表和保存最终模型 ---")
        # (补丁 6, 9)
        save_training_data_safely(tracking_data, H.OUTPUT_DIR)

        if hdqn_net is not None:
            hdqn_net.saveWeight(str(H.OUTPUT_DIR / "sfc_hirl_model_final"))
            logger.info("最终模型已保存。")

        logger.info("--- 分析完成, 清理资源 ---")


# ============================================
# 修复: 添加调试和可视化工具
# ============================================

def visualize_state(env):
    """可视化当前环境状态（调试用）"""
    print(f"\n=== 当前状态 (T={env.t}) ===")
    if env.current_request:
        print(f"请求 ID: {env.current_request['id']}")
        print(f"源: {env.current_request['source']}, 目的: {env.current_request['dest']}")
        print(f"未完成目的: {env.unadded_dest_indices}")
        print(f"树上节点数: {len(env.nodes_on_tree)}")
        print(f"树上路径数: {len(env.current_tree['paths_map']) if env.current_tree else 0}")
    print(f"平均CPU利用率: {1 - np.mean(env.C / env.C_cap):.2%}")
    print(f"平均带宽利用率: {1 - np.mean(env.B / env.B_cap):.2%}")
    print(f"已服务请求数: {len(env.served_requests)}")
    print("=" * 40)


def validate_action_space(env, action):
    """验证动作是否有效（调试用）"""
    valid_actions = env.get_valid_low_level_actions()
    if action not in valid_actions:
        # (使用 logging, 但保留 print 以便在调试时更显眼)
        print(f"警告: 动作 {action} 不在有效动作集 {valid_actions} 中")  # (Emoji removed)
        return False
    return True


if __name__ == "__main__":
    main_improved()  # (补丁 8)