#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_hirl_sfc_v14_commented.py
集成详细的 Expert/Agent 决策对比、低层动作解码及最终路径树日志。
已添加全量中文注释。
"""
from __future__ import annotations
import os
import sys

# [CRITICAL FIX] 在导入 pyplot 之前强制使用非交互式后端
# 这对于在服务器或无显示器的环境中运行至关重要，防止 "Tcl/Tk" 错误
os.environ['MPLBACKEND'] = 'Agg'
try:
    import sys

    # Hack: 假装没有安装 IPython，以绕过 matplotlib 对 IPython 版本检测的 Bug
    sys.modules['IPython'] = None
except ImportError:
    pass

import logging
import traceback
from collections import namedtuple
from pathlib import Path
import numpy as np
import pandas as pd

# 再次确保 Matplotlib 使用 Agg 后端（用于生成文件而非显示窗口）
import matplotlib

matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from tensorflow.keras.utils import to_categorical

# 导入项目自定义模块
import hyperparameters as H
from hirl_sfc_env import SFC_HIRL_Env
from hirl_sfc_models import MetaControllerNN, Hdqn_SFC
from hirl_sfc_agent import Agent_SFC

# 定义经验回放的数据结构
ActorExperience = namedtuple("ActorExperience", ["state", "goal", "action", "reward", "next_state", "done"])

# 配置日志格式：输出到控制台
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[
                        logging.StreamHandler(sys.stdout)
                    ])
logger = logging.getLogger(__name__)


def safe_update_agent(agent, t: int = None):
    """
    尝试以兼容的方式调用 Agent 的训练/更新函数。

    由于不同的 Agent 实现可能使用不同的方法名（如 update, train, train_step），
    此函数会依次尝试这些名称，并根据函数签名决定是否传递时间步参数 `t`。

    参数:
        agent: 智能体对象
        t (int, optional): 当前的时间步或训练步数

    返回:
        调用结果（通常是 loss 等指标），如果未找到兼容方法则返回 None。
    """
    try_names = ['update', 'train_from_memory', 'train', 'train_step']
    for name in try_names:
        fn = getattr(agent, name, None)
        if callable(fn):
            try:
                # 检查函数签名，看是否需要参数
                import inspect
                sig = inspect.signature(fn)
                if len(sig.parameters) == 0:
                    return fn()
                else:
                    return fn(t)
            except Exception:
                # 如果当前方法调用失败，尝试下一个
                continue
    return None


def safe_store_and_maybe_train(agent, exp: ActorExperience, step_count: int):
    """
    将经验存储到 Replay Buffer，并根据 Agent 的训练频率触发训练。

    参数:
        agent: 智能体对象
        exp (ActorExperience): 要存储的经验元组 (s, g, a, r, s', d)
        step_count (int): 当前总步数，用于判断是否达到训练频率
    """
    # 1. 尝试存储经验
    try:
        if hasattr(agent, 'store'):
            agent.store(exp)
        elif hasattr(agent, 'memory_push'):
            agent.memory_push(exp)
    except Exception:
        pass

    # 2. 获取训练频率配置
    train_freq = getattr(agent, 'trainFreq', getattr(agent, 'train_freq', getattr(agent, 'trainFreq', None)))

    # 3. 触发训练
    if train_freq is None:
        # 如果没有定义频率，默认每步尝试更新
        safe_update_agent(agent, step_count)
    else:
        # 按照频率更新
        if train_freq > 0 and (step_count % train_freq == 0):
            safe_update_agent(agent, step_count)


def save_training_plots_from_df(df: pd.DataFrame, out_dir: Path):
    """
    根据训练数据 DataFrame 生成并保存可视化图表。
    使用面向对象的 Matplotlib API 以避免线程冲突和 GUI 错误。

    参数:
        df (pd.DataFrame): 包含训练指标的数据表
        out_dir (Path): 图片保存目录
    """
    try:
        out_dir.mkdir(parents=True, exist_ok=True)

        def smoothed(values, window=100):
            """辅助函数：计算滑动平均值"""
            if len(values) == 0: return values
            window = min(window, max(1, len(values)))
            return pd.Series(values).rolling(window, min_periods=1).mean().values

        def plot_column(col, title, ylabel, smooth=False):
            """辅助函数：绘制单列数据"""
            if col not in df.columns: return

            # 创建画布
            fig = Figure(figsize=(10, 6), dpi=100)
            ax = fig.add_subplot(1, 1, 1)

            data = df[col].values
            # 绘制原始数据（浅色）
            ax.plot(data, alpha=0.3, color='blue', label='Raw')
            # 绘制平滑数据（深色）
            if smooth:
                ax.plot(smoothed(data), color='red', linewidth=2, label='Smoothed')

            ax.set_title(title)
            ax.set_xlabel("Episode")
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 保存文件
            fname = out_dir / f"{col}.png"
            FigureCanvas(fig).print_figure(str(fname))
            logger.info("Saved plot %s", fname)

        # 绘制关键指标
        plot_column('reward', 'Episode Reward', 'Reward', smooth=True)
        plot_column('acceptance_rate', 'Request Acceptance Rate', '%', smooth=True)
        plot_column('blocking_rate', 'Blocking Rate', '%', smooth=True)
        plot_column('avg_cpu_util', 'Avg CPU Utilization', '%', smooth=True)
        plot_column('avg_bw_util', 'Avg BW Utilization', '%', smooth=True)

    except Exception:
        logger.exception("绘图失败")


def save_tracking_data(tracking_data: dict, out_dir: Path):
    """
    保存训练追踪数据到 CSV 文件，并调用绘图函数。

    参数:
        tracking_data (dict): 包含各项指标列表的字典
        out_dir (Path): 输出目录
    """
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(tracking_data)
        csv_path = out_dir / "training_metrics.csv"
        df.to_csv(csv_path, index=False)
        save_training_plots_from_df(df, out_dir)
    except Exception:
        logger.exception("保存训练数据失败")


def main():
    """
    主训练循环。
    包含初始化、预训练（模仿学习）、正式训练（混合强化学习+DAgger）及结果保存。
    """
    # 1. 检查输入数据
    if not H.INPUT_DIR.exists():
        print(f"输入数据目录不存在: {H.INPUT_DIR}")
        return

    # 2. 配置输出目录和文件日志
    H.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(H.OUTPUT_DIR / "training.log", encoding='utf-8')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    logger.info("启动 HIRL-SFC 训练（V14 Full Logging）")

    # 初始化指标追踪字典
    tracking_data = {
        'episode': [], 'reward': [],
        'acceptance_rate': [], 'blocking_rate': [],
        'avg_cpu_util': [], 'avg_mem_util': [], 'avg_bw_util': []
    }

    # 3. 初始化环境、网络模型和智能体
    env = SFC_HIRL_Env(H.INPUT_DIR, H.TOPOLOGY_MATRIX, H.DC_NODES, H.CAPACITIES)
    STATE_SHAPE = env.observation_space.shape
    NB_GOALS = env.NB_HIGH_LEVEL_GOALS
    NB_ACTIONS = env.NB_LOW_LEVEL_ACTIONS

    # 元控制器（高层策略）
    metacontroller = MetaControllerNN(state_shape=STATE_SHAPE, n_goals=NB_GOALS, lr=H.LR)
    # 低层 DQN 网络
    hdqn_net = Hdqn_SFC(state_shape=STATE_SHAPE, n_goals=NB_GOALS, n_actions=NB_ACTIONS, lr=H.LR)

    # 低层智能体
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

    logger.info("初始化完成: STATE_SHAPE=%s NB_GOALS=%d NB_ACTIONS=%d", STATE_SHAPE, NB_GOALS, NB_ACTIONS)

    # ==========================================
    # 阶段 1：模仿学习预训练 (Imitation Learning)
    # ==========================================
    logger.info("阶段1：模仿学习预训练，步数=%d", H.PRE_TRAIN_STEPS)
    t = 0
    current_request, high_level_state = env.reset_request()

    while t < H.PRE_TRAIN_STEPS and current_request is not None:
        try:
            # 获取专家解决方案（直接计算出一条可行路径）
            try:
                _, expert_traj = env.expert.solve_request_for_expert(current_request, env._get_network_state_dict())
            except Exception:
                expert_traj = None

            # 如果专家也无法解决，跳过此请求
            if not expert_traj:
                t += 1
                current_request, high_level_state = env.reset_request()
                continue

            # 遍历专家轨迹，将其转化为经验存入 Buffer
            for (exp_goal, exp_action_tuple, exp_cost) in expert_traj:
                # 如果该目标已在环境中完成，则跳过
                if exp_goal not in env.unadded_dest_indices: continue

                # 将专家动作转换为动作索引
                exp_action = exp_action_tuple[0] * env.K_path + exp_action_tuple[1]

                # 在环境中执行一步（State -> Next State）
                next_state, cost, sub_task_done, request_done = env.step_low_level(exp_goal, exp_action)

                # 构建监督信号 Reward
                reward = 1.0 if sub_task_done else -1.0
                goal_one_hot = to_categorical(exp_goal, num_classes=NB_GOALS)

                # 存储经验
                experience = ActorExperience(high_level_state, goal_one_hot, exp_action, reward, next_state,
                                             sub_task_done)
                safe_store_and_maybe_train(low_level_agent, experience, t)

                # 同时也训练高层元控制器（Meta Controller）学习专家的目标选择
                try:
                    metacontroller.collect(high_level_state, exp_goal)
                    if t % H.META_TRAIN_FREQ == 0 and metacontroller.check_training_clock():
                        metacontroller.train()
                except Exception:
                    pass

                high_level_state = next_state
                t += 1

                # 检查预训练是否结束
                if t >= H.PRE_TRAIN_STEPS or request_done: break

            # 获取下一个请求
            current_request, high_level_state = env.reset_request()

        except Exception:
            # 容错：跳过当前请求
            t += 1
            current_request, high_level_state = env.reset_request()

    logger.info("预训练结束")

    # ==========================================
    # 阶段 2：混合 IL / RL 训练 (Mixed Training)
    # ==========================================
    logger.info("阶段2：混合 IL/RL 训练开始")
    step_count = t
    episode_count = 0

    # DAgger 衰减参数：随着训练进行，减少对专家的依赖
    dagger_total = max(1, H.STEPS_LIMIT)
    dagger_initial = 1.0
    dagger_final = 0.05

    while episode_count < H.EPISODE_LIMIT and step_count < H.STEPS_LIMIT:
        try:
            # 重置环境获取新请求
            current_request, high_level_state = env.reset_request()
            if current_request is None: break  # 数据集耗尽

            episode_reward = 0.0
            episode_steps = 0
            request_done = False
            req_id = current_request.get('id', 'unknown')

            # 开始处理一个请求（Episode）
            while not request_done:
                high_state_v = np.reshape(high_level_state, (1, -1))

                # --- 1. 元控制器预测 (Agent Policy) ---
                try:
                    goal_probs = metacontroller.predict(high_state_v)
                    agent_chosen_goal = metacontroller.sample(goal_probs)
                except Exception:
                    # Fallback: 选择第一个未完成的目标
                    agent_chosen_goal = list(env.unadded_dest_indices)[0] if env.unadded_dest_indices else 0

                # >>> 新增日志 (2): 记录 Agent 想要去哪个节点
                try:
                    dest_list = current_request.get('dest', [])
                    agent_node = dest_list[agent_chosen_goal] if agent_chosen_goal < len(dest_list) else None
                    logger.info(f"[Agent] Policy recommends goal_index={agent_chosen_goal}, node={agent_node}")
                except Exception:
                    pass

                # --- 2. 获取专家意见 (Expert Policy) ---
                true_goal = env.get_expert_high_level_goal(high_state_v)

                # >>> 新增日志 (1): 记录 Expert 推荐去哪个节点
                try:
                    dest_list = current_request.get('dest', [])
                    expert_node = dest_list[true_goal] if true_goal < len(dest_list) else None
                    logger.info(f"[Expert] Recommend goal_index={true_goal}, node={expert_node}")
                except Exception:
                    pass

                # --- 3. DAgger 决策 (Expert vs Agent) ---
                # 根据当前步数计算 beta 值，概率性地使用专家指导
                beta = dagger_final + (dagger_initial - dagger_final) * max(0, 1 - (step_count / dagger_total))
                use_expert = (np.random.rand() < beta)
                chosen_goal = true_goal if use_expert else agent_chosen_goal

                # >>> 新增日志 (3): 记录最终采用了谁的决策
                if chosen_goal == true_goal and use_expert:
                    logger.info(f"   [Decision] Expert USED for goal_index={true_goal}")
                elif chosen_goal != true_goal:
                    logger.info(f"   [Decision] Agent USED goal_index={chosen_goal} | Expert would be {true_goal}")

                # 有效性检查：如果选中的目标已完成，强制使用专家，若专家也无效则退出
                if chosen_goal not in env.unadded_dest_indices:
                    # logger.warning(f"   [Warn] Goal {chosen_goal} invalid. Switching to Expert {true_goal}")
                    chosen_goal = true_goal
                    if chosen_goal not in env.unadded_dest_indices: break

                # --- 4. 低层执行循环 (Low Level Execution) ---
                sub_task_done = False
                low_level_state = high_level_state
                attempts = 0

                # 尝试多次低层动作直到子任务完成或超时
                while not sub_task_done and attempts < 15:
                    attempts += 1
                    low_v = np.reshape(low_level_state, (1, -1))
                    valid_actions = env.get_valid_low_level_actions()

                    # RL Agent 选择动作
                    try:
                        action = low_level_agent.selectMove(low_v,
                                                            np.reshape(to_categorical(chosen_goal, NB_GOALS), (1, -1)),
                                                            valid_actions)
                    except Exception:
                        action = valid_actions[0]

                    # >>> 新增日志 (4): 动作解码与预期路径
                    # 尝试将抽象的 action ID 解析为人类可读的路径信息
                    try:
                        if hasattr(env, '_decode_low_level_action'):
                            i_idx, k_idx = env._decode_low_level_action(int(action))
                        else:
                            k_idx = int(action % getattr(env, 'K_path', 1))
                            i_idx = int(action // getattr(env, 'K_path', 1))

                        parsed_path = None
                        try:
                            if hasattr(env, 'path_manager'):
                                parsed_path = env.path_manager.get_path(i_idx)
                        except:
                            pass
                        if parsed_path is None:
                            dest_nodes = current_request['dest']
                            dest_node_val = dest_nodes[chosen_goal]
                            parsed_path = env.current_tree.get('paths_map', {}).get(dest_node_val, None)

                        logger.info(
                            f"   [Low-level] action={action}, decoded=(i_idx={i_idx},k_idx={k_idx}), parsed_path={parsed_path}")
                    except:
                        logger.info(f"   [Low-level] action={action} (decode failed)")

                    # 执行环境步
                    next_state, cost, sub_task_done, request_done = env.step_low_level(chosen_goal, action)

                    # >>> 新增日志 (5): 子任务完成时打印最终路径
                    if sub_task_done:
                        try:
                            dest_node_idx = current_request['dest'][chosen_goal]
                            final_path = env.current_tree.get('paths_map', {}).get(dest_node_idx, [])
                            logger.info(
                                f"   >>> [Success] Goal {chosen_goal} Reached. dest_node={dest_node_idx}, Path={final_path}")
                        except:
                            pass

                    # 计算 Reward
                    try:
                        reward = low_level_agent.criticize(sub_task_done, cost, request_failed=(not sub_task_done))
                    except Exception:
                        reward = 1.0 if sub_task_done else -1.0

                    # 存储经验
                    exp = ActorExperience(high_level_state, to_categorical(chosen_goal, NB_GOALS).flatten(), action,
                                          reward, next_state, sub_task_done)
                    safe_store_and_maybe_train(low_level_agent, exp, step_count)

                    # 无论是否使用了专家，都收集专家数据用于元控制器训练 (DAgger Aggregation)
                    try:
                        metacontroller.collect(high_level_state, true_goal)
                        if step_count % H.META_TRAIN_FREQ == 0 and metacontroller.check_training_clock():
                            metacontroller.train()
                    except Exception:
                        pass

                    low_level_state = next_state
                    step_count += 1
                    episode_steps += 1
                    episode_reward += reward

                    if request_done: break

                # 更新高层状态
                high_level_state = low_level_state

            # >>> 新增日志 (6): 请求完成后打印整棵组播树
            try:
                tree_map = env.current_tree.get('paths_map', {})
                logger.info(f"[Req {req_id}] Final multicast paths_map: {tree_map}")
            except:
                pass

            # --- Episode 统计数据计算 ---
            avg_cpu = (1.0 - np.mean(env.C) / env.C_cap) * 100.0
            avg_bw = (1.0 - np.mean(env.B) / env.B_cap) * 100.0

            tracking_data['episode'].append(episode_count)
            tracking_data['reward'].append(episode_reward)

            total_req = max(1, getattr(env, 'total_requests_seen', 1))
            acc_rate = (getattr(env, 'total_requests_accepted', 0) / total_req) * 100.0
            blk_rate = 100.0 - acc_rate

            tracking_data['acceptance_rate'].append(acc_rate)
            tracking_data['blocking_rate'].append(blk_rate)
            tracking_data['avg_cpu_util'].append(avg_cpu)
            tracking_data['avg_mem_util'].append((1.0 - np.mean(env.M) / env.M_cap) * 100.0)
            tracking_data['avg_bw_util'].append(avg_bw)

            logger.info(
                f"Episode {episode_count}: Reward={episode_reward:.2f}, Steps={episode_steps}, Acc={acc_rate:.1f}%, BW={avg_bw:.1f}%")

            episode_count += 1
            # 每 50 回合保存一次模型
            if episode_count % 50 == 0:
                hdqn_net.saveWeight(str(H.OUTPUT_DIR / f"model_ep{episode_count}"))

        except KeyboardInterrupt:
            logger.warning("用户中断训练")
            break
        except Exception:
            logger.exception("Episode failed")
            episode_count += 1

    # 4. 训练结束
    logger.info("Training finished.")
    save_tracking_data(tracking_data, H.OUTPUT_DIR)
    if hdqn_net:
        hdqn_net.saveWeight(str(H.OUTPUT_DIR / "model_final"))


if __name__ == "__main__":
    main()