# -*- coding: utf-8 -*-
# @File    : env_sfc.py
# @Author  : 郭旭翔研究环境整合完整版
# @Desc    : 基于强化学习的多播 SFC 映射环境（容错性增强稳定版）

import numpy as np
import networkx as nx
import torch
import warnings
import logging
from collections import defaultdict
import heapq
import pickle


class SFCEnv:
    """
    基于强化学习的多播 SFC 部署环境
    —— 支持多播树扩展、资源约束与 QoS 优化
    —— 具备完善的容错处理和边界检查
    """

    def __init__(self, graph, vnfs, conf):
        # 初始化日志
        self._init_logging()

        self.graph = graph
        self.vnfs = vnfs
        self.conf = conf

        # 验证输入数据
        self._validate_inputs()

        self.nodes = list(self.graph.nodes())
        self.edges = list(self.graph.edges())

        # 建立节点索引映射（保证动作空间稳定）
        self.index_to_node = {i: n for i, n in enumerate(self.nodes)}
        self.node_to_index = {n: i for i, n in enumerate(self.nodes)}

        # 节点与边的初始资源备份（带容错）
        self.initial_node_cpu = self._init_node_cpu_backup()
        self.initial_edge_bandwidth = self._init_edge_bandwidth_backup()

        # 环境状态变量
        self.start = None
        self.tree_nodes = set()
        self.route_graph = nx.Graph()
        self.current_vnf_index = 0
        self.done = False
        self.used_links = defaultdict(float)
        self.vnf_deployment = {}

        # 配置参数校验
        self._validate_conf(conf)

        # 奖励权重系数
        self.beta1, self.beta2, self.beta3, self.beta4, self.beta5 = 1.0, 1.0, 1.0, 0.3, 0.2
        self.delta_qos_penalty = 0.8  # QoS下降惩罚系数

        # 初始化 QoS 矩阵
        self._init_graph_parameters()

        # 定义动作空间和状态形状（动态计算）
        self.action_space = list(range(len(self.nodes)))
        self.state_shape = self._compute_state_shape()

        # 上一步 QoS 记录（用于奖励增益）
        self.prev_global_qos = 0

        # 优化配置参数
        self.enable_multi_path_selection = getattr(conf, 'enable_multi_path_selection', False)
        self.max_candidate_paths = getattr(conf, 'max_candidate_paths', 3)

        # QoS归一化基准值
        self.max_delay = self._get_max_delay()
        self.max_bandwidth = self._get_max_bandwidth()

        self.logger.info(
            f"环境初始化完成 - 节点数: {len(self.nodes)}, 边数: {len(self.edges)}, VNF数: {len(self.vnfs)}")

    def _init_logging(self):
        """初始化日志系统"""
        self.logger = logging.getLogger('SFCEnv')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _validate_inputs(self):
        """验证输入数据的完整性"""
        if not self.graph or len(self.graph.nodes()) == 0:
            raise ValueError("网络图不能为空")

        if not self.vnfs:
            raise ValueError("VNF列表不能为空")

        # 检查VNF必需的CPU属性
        for i, vnf in enumerate(self.vnfs):
            if 'cpu' not in vnf:
                raise ValueError(f"VNF {i} 缺少CPU属性")
            if vnf['cpu'] <= 0:
                raise ValueError(f"VNF {i} 的CPU需求必须为正数")

    def _init_node_cpu_backup(self):
        """初始化节点CPU备份（带容错）"""
        cpu_backup = {}
        for n in self.nodes:
            cpu = self.graph.nodes[n].get('cpu', 1.0)
            cpu_total = self.graph.nodes[n].get('cpu_total', max(cpu, 1.0))
            # 确保cpu_total不小于cpu
            cpu_total = max(cpu_total, cpu)
            self.graph.nodes[n]['cpu'] = cpu
            self.graph.nodes[n]['cpu_total'] = cpu_total
            cpu_backup[n] = cpu
        return cpu_backup

    def _init_edge_bandwidth_backup(self):
        """初始化边带宽备份（带容错）"""
        edge_bandwidth = {}
        for u, v in self.graph.edges():
            key = tuple(sorted((u, v)))
            bandwidth = self.graph[u][v].get('bandwidth', 1.0)
            # 设置默认的delay和loss
            self.graph[u][v]['delay'] = self.graph[u][v].get('delay', 10.0)
            self.graph[u][v]['loss'] = self.graph[u][v].get('loss', 0.01)
            self.graph[u][v]['bandwidth'] = bandwidth
            edge_bandwidth[key] = bandwidth
        return edge_bandwidth

    def _get_max_delay(self):
        """获取最大延迟（带容错）"""
        try:
            delays = nx.get_edge_attributes(self.graph, 'delay').values()
            return max(delays) if delays else 100.0
        except:
            return 100.0

    def _get_max_bandwidth(self):
        """获取最大带宽（带容错）"""
        try:
            bandwidths = nx.get_edge_attributes(self.graph, 'bandwidth').values()
            return max(bandwidths) if bandwidths else 1000.0
        except:
            return 1000.0

    def _compute_state_shape(self):
        """动态计算状态形状"""
        n = len(self.nodes)
        return (7, n, n)

    # ----------------------------------------------------------
    # 配置参数校验
    # ----------------------------------------------------------
    def _validate_conf(self, conf):
        """校验配置参数合法性"""
        # 检查带宽请求（必须为正数）
        self.bandwidth_request = getattr(conf, 'bandwidth_request', 1.0)
        if self.bandwidth_request <= 0:
            warnings.warn(f"bandwidth_request={self.bandwidth_request} 非法，强制设为1.0")
            self.bandwidth_request = 1.0

        # 检查路径cutoff（必须为正整数）
        self.path_cutoff = getattr(conf, 'path_cutoff', 4)
        if not isinstance(self.path_cutoff, int) or self.path_cutoff <= 0:
            warnings.warn(f"path_cutoff={self.path_cutoff} 非法，强制设为4")
            self.path_cutoff = 4

        # 检查奖励权重
        reward_weights = ['beta1', 'beta2', 'beta3', 'beta4', 'beta5']
        for weight in reward_weights:
            if hasattr(conf, weight):
                value = getattr(conf, weight)
                if value < 0:
                    warnings.warn(f"{weight}={value} 为负值，可能影响奖励计算")

    # ----------------------------------------------------------
    # 初始化图参数矩阵
    # ----------------------------------------------------------
    def _init_graph_parameters(self):
        """初始化图参数矩阵（带容错）"""
        try:
            bw_matrix = nx.to_numpy_array(self.graph, weight='bandwidth')
            delay_matrix = nx.to_numpy_array(self.graph, weight='delay')
            loss_matrix = nx.to_numpy_array(self.graph, weight='loss')

            self.normal_bw_matrix = np.clip(bw_matrix / (bw_matrix.max() + 1e-6), 0, 1)
            self.normal_delay_matrix = np.clip(1 - (delay_matrix / (delay_matrix.max() + 1e-6)), 0, 1)
            self.normal_loss_matrix = np.clip(1 - (loss_matrix / (loss_matrix.max() + 1e-6)), 0, 1)

            # 为图添加复合权重，用于路径查找
            for u, v in self.graph.edges():
                bw_norm = self.graph[u][v].get('bandwidth', 1.0) / (bw_matrix.max() + 1e-6)
                delay_norm = 1 - (self.graph[u][v].get('delay', 10.0) / (delay_matrix.max() + 1e-6))
                loss_norm = 1 - (self.graph[u][v].get('loss', 0.01) / (loss_matrix.max() + 1e-6))
                composite = 0.4 * bw_norm + 0.4 * delay_norm + 0.2 * loss_norm
                self.graph[u][v]['composite_weight'] = 1 - composite  # 转换为成本
        except Exception as e:
            self.logger.warning(f"图参数初始化失败: {e}, 使用默认值")
            n = len(self.nodes)
            self.normal_bw_matrix = np.ones((n, n)) * 0.5
            self.normal_delay_matrix = np.ones((n, n)) * 0.5
            self.normal_loss_matrix = np.ones((n, n)) * 0.5

    # ----------------------------------------------------------
    # 环境重置
    # ----------------------------------------------------------
    def reset(self):
        """环境重置（带完整容错）"""
        self.logger.info("重置环境")

        self.done = False
        self.current_vnf_index = 0
        self.tree_nodes = set()
        self.route_graph = nx.Graph()
        self.used_links = defaultdict(float)
        self.vnf_deployment = {}
        self.prev_global_qos = 0

        # 恢复所有节点 CPU
        for n in self.nodes:
            self.graph.nodes[n]['cpu'] = self.initial_node_cpu[n]

        # 恢复边带宽
        for (u, v), bw in self.initial_edge_bandwidth.items():
            if self.graph.has_edge(u, v):
                self.graph[u][v]['bandwidth'] = bw

        # 选择起点（带完整验证）
        valid_starts = []
        for n in self.nodes:
            cpu_req = self.vnfs[0]['cpu'] if self.vnfs else 1.0
            node_cpu = self.graph.nodes[n].get('cpu', 0.0)
            if node_cpu >= cpu_req:
                valid_starts.append(n)

        if not valid_starts:
            self.logger.warning("无可用起始节点，选择CPU资源最多的节点")
            valid_starts = sorted(self.nodes,
                                  key=lambda n: self.graph.nodes[n].get('cpu', 0.0),
                                  reverse=True)[:1]
            if valid_starts:
                # 强制调整资源以满足需求
                node = valid_starts[0]
                cpu_req = self.vnfs[0]['cpu'] if self.vnfs else 1.0
                current_cpu = self.graph.nodes[node].get('cpu', 0.0)
                if current_cpu < cpu_req:
                    self.graph.nodes[node]['cpu'] = cpu_req
                    self.logger.warning(f"强制调整节点 {node} 的CPU资源: {current_cpu} -> {cpu_req}")

        if not valid_starts:
            raise ValueError("无法找到合适的起始节点")

        self.start = np.random.choice(valid_starts)
        self.tree_nodes.add(self.start)
        self.route_graph.add_node(self.start)

        # 部署第一个 VNF
        if self.vnfs:
            cpu_req = self.vnfs[0]['cpu']
            current_cpu = self.graph.nodes[self.start].get('cpu', 0.0)
            self.graph.nodes[self.start]['cpu'] = max(current_cpu - cpu_req, 0.0)
            self.vnf_deployment[0] = self.start
            self.current_vnf_index = 1

        self.logger.info(f"起始节点: {self.start}, 剩余VNF: {len(self.vnfs) - self.current_vnf_index}")

        return self._get_state_matrix(), {}

    # ----------------------------------------------------------
    # 状态矩阵构造（七通道）- 容错增强版
    # ----------------------------------------------------------
    def _get_state_matrix(self):
        """构造状态矩阵（带完整容错）"""
        n = len(self.nodes)
        cpu_matrix = np.zeros((n, n))
        bw_matrix = np.zeros((n, n))
        delay_matrix = np.zeros((n, n))
        deployment_matrix = np.zeros((n, n))
        tree_topology_matrix = np.zeros((n, n))
        link_usage_matrix = np.zeros((n, n))
        progress_matrix = np.zeros((n, n))

        for i, node_i in enumerate(self.nodes):
            for j, node_j in enumerate(self.nodes):
                if i == j:
                    # 节点自身属性（带容错）
                    cpu_total = self.graph.nodes[node_i].get('cpu_total', 1.0)
                    cpu_used = self.graph.nodes[node_i].get('cpu', 0.0)
                    cpu_ratio = np.clip(cpu_used / max(cpu_total, 1e-6), 0, 1)
                    cpu_matrix[i][j] = cpu_ratio

                    deployment_matrix[i][j] = 1.0 if node_i in self.vnf_deployment.values() else 0.0

                    # 部署进度
                    progress = self.current_vnf_index / max(len(self.vnfs), 1)
                    progress_matrix[i][j] = progress
                else:
                    # 边属性（带容错）
                    if self.graph.has_edge(node_i, node_j):
                        idx_i, idx_j = self.node_to_index[node_i], self.node_to_index[node_j]
                        bw_matrix[i][j] = self.normal_bw_matrix[idx_i][idx_j] if idx_i < len(
                            self.nodes) and idx_j < len(self.nodes) else 0.5
                        delay_matrix[i][j] = self.normal_delay_matrix[idx_i][idx_j] if idx_i < len(
                            self.nodes) and idx_j < len(self.nodes) else 0.5

                        if self.route_graph.has_edge(node_i, node_j):
                            tree_topology_matrix[i][j] = 1.0

                        # 链路占用比例
                        link_key = tuple(sorted((node_i, node_j)))
                        total_bw = self.initial_edge_bandwidth.get(link_key, 1.0)
                        used_bw = self.used_links.get(link_key, 0.0)
                        usage_ratio = np.clip(used_bw / max(total_bw, 1e-6), 0, 1)
                        link_usage_matrix[i][j] = usage_ratio

        # 7通道状态矩阵
        state = np.stack([
            cpu_matrix, bw_matrix, delay_matrix,
            deployment_matrix, tree_topology_matrix,
            link_usage_matrix, progress_matrix
        ], axis=0)

        return torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    # ----------------------------------------------------------
    # 执行动作（完整边界检查）
    # ----------------------------------------------------------
    def step(self, action):
        """执行动作（带完整边界检查）"""
        if self.done:
            return self._get_state_matrix(), 0.0, True, {"termination": "already_done"}

        # 动作边界检查
        if not (0 <= action < len(self.action_space)):
            self.logger.warning(f"动作越界: {action}, 动作空间: [0, {len(self.action_space) - 1}]")
            return self._get_state_matrix(), -5.0, False, {"error": "动作越界"}

        node = self.index_to_node[int(action)]

        # 检查是否有有效动作
        if not self._has_valid_actions():
            self.done = True
            return self._get_state_matrix(), -10.0, True, {"termination": "no_valid_actions"}

        # 获取当前VNF需求
        if self.current_vnf_index >= len(self.vnfs):
            self.done = True
            return self._get_state_matrix(), 0.0, True, {"termination": "sfc_completed"}

        cpu_req = self.vnfs[self.current_vnf_index]['cpu']

        # 验证动作有效性
        is_valid, msg = self.validate_action(node)
        if not is_valid:
            self.logger.warning(f"无效动作: {action} -> {node}, 原因: {msg}")
            return self._get_state_matrix(), -5.0, False, {"error": msg}

        # 执行部署逻辑
        if node in self.tree_nodes:
            # 节点已在树中
            current_cpu = self.graph.nodes[node].get('cpu', 0.0)
            self.graph.nodes[node]['cpu'] = max(current_cpu - cpu_req, 0.0)
            reward = self._calculate_sfc_reward([], node, cpu_req)
            path_used = []
            self.logger.info(f"节点 {node} 已在树中，直接部署VNF {self.current_vnf_index}")
        else:
            # 需要连接新节点
            best_path = self._find_multi_objective_path_to_tree(node)
            if best_path is None:
                self.logger.warning(f"无法找到从多播树到节点 {node} 的路径")
                return self._get_state_matrix(), -8.0, False, {"error": "无可行路径"}

            if not self._check_path_bandwidth(best_path):
                self.logger.warning(f"路径 {best_path} 带宽不足")
                return self._get_state_matrix(), -6.0, False, {"error": "带宽不足"}

            new_links = self._add_path_to_tree(best_path)
            self.tree_nodes.update(set(best_path))  # 使用set去重
            current_cpu = self.graph.nodes[node].get('cpu', 0.0)
            self.graph.nodes[node]['cpu'] = max(current_cpu - cpu_req, 0.0)
            self._deduct_path_bandwidth(new_links)
            reward = self._calculate_sfc_reward(best_path, node, cpu_req)
            path_used = best_path
            self.logger.info(f"通过路径 {best_path} 连接节点 {node}，部署VNF {self.current_vnf_index}")

        # 记录部署
        self.vnf_deployment[self.current_vnf_index] = node
        self.current_vnf_index += 1

        # 检查终止条件
        info = self._check_termination(reward)

        info.update({
            "vnf_index": self.current_vnf_index - 1,
            "deployed_node": node,
            "path_used": path_used,
            "tree_size": len(self.tree_nodes),
            "reward": reward
        })

        return self._get_state_matrix(), reward, self.done, info

    def _check_termination(self, reward):
        """检查终止条件"""
        if self.current_vnf_index >= len(self.vnfs):
            self.done = True
            return {"termination": "sfc_completed"}
        elif not self._has_valid_actions():
            self.done = True
            reward -= 8.0  # 提前终止惩罚
            return {"termination": "next_vnf_no_valid_actions"}
        else:
            return {"termination": "in_progress"}

    # ----------------------------------------------------------
    # 多目标路径选择（容错增强版）
    # ----------------------------------------------------------
    def _find_multi_objective_path_to_tree(self, target_node):
        """路径选择（带完整容错）"""
        candidate_paths = []

        for tree_node in self.tree_nodes:
            if tree_node == target_node:
                continue

            try:
                if self.enable_multi_path_selection:
                    paths = self._k_shortest_paths(tree_node, target_node, self.max_candidate_paths)
                else:
                    path = nx.shortest_path(self.graph, tree_node, target_node, weight="composite_weight")
                    paths = [path] if len(path) <= self.path_cutoff else []

                for path in paths:
                    if len(path) > self.path_cutoff:
                        continue

                    # 计算路径指标（带容错）
                    try:
                        delay = nx.path_weight(self.graph, path, weight="delay")
                    except:
                        delay = 50.0  # 默认延迟

                    normalized_delay = np.clip(1 - (delay / self.max_delay), 0, 1)

                    # 使用归一化带宽矩阵
                    normalized_bw = 0.5  # 默认值
                    try:
                        bw_values = []
                        for u, v in zip(path[:-1], path[1:]):
                            if self.graph.has_edge(u, v):
                                idx_u = self.node_to_index.get(u, 0)
                                idx_v = self.node_to_index.get(v, 0)
                                if idx_u < len(self.nodes) and idx_v < len(self.nodes):
                                    bw_values.append(self.normal_bw_matrix[idx_u][idx_v])
                        normalized_bw = np.min(bw_values) if bw_values else 0.5
                    except:
                        pass

                    # 计算丢包率（带容错）
                    try:
                        loss = 1 - np.prod([1 - self.graph[u][v].get('loss', 0.01)
                                            for u, v in zip(path[:-1], path[1:])])
                        normalized_loss = np.clip(1 - loss, 0, 1)
                    except:
                        normalized_loss = 0.9

                    if self.enable_multi_path_selection:
                        # 考虑节点资源
                        cpu_ratios = []
                        for n in path:
                            cpu_total = self.graph.nodes[n].get('cpu_total', 1.0)
                            cpu_used = self.graph.nodes[n].get('cpu', 0.0)
                            cpu_ratio = np.clip(cpu_used / max(cpu_total, 1e-6), 0, 1)
                            cpu_ratios.append(cpu_ratio)
                        avg_cpu_ratio = np.mean(cpu_ratios) if cpu_ratios else 0.5

                        score = (0.3 * normalized_delay +
                                 0.3 * normalized_bw +
                                 0.2 * normalized_loss +
                                 0.2 * avg_cpu_ratio)
                    else:
                        score = (0.4 * normalized_delay +
                                 0.4 * normalized_bw +
                                 0.2 * normalized_loss)

                    candidate_paths.append((path, score))

            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

        if not candidate_paths:
            self.logger.warning(f"目标节点 {target_node} 与多播树无可用路径")
            return None

        return max(candidate_paths, key=lambda x: x[1])[0]

    def _k_shortest_paths(self, source, target, k):
        """K最短路径算法"""
        try:
            paths = nx.shortest_simple_paths(self.graph, source, target, weight="composite_weight")
            result = []
            for _ in range(k):
                try:
                    result.append(next(paths))
                except StopIteration:
                    break
            return result
        except:
            return []

    # ----------------------------------------------------------
    # 带宽检查与扣除（容错增强版）
    # ----------------------------------------------------------
    def _check_path_bandwidth(self, path):
        """检查路径带宽（带容错）"""
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if not self.graph.has_edge(u, v):
                return False

            link_key = tuple(sorted((u, v)))
            current_bandwidth = self.graph[u][v].get('bandwidth', 1.0)
            total_used = self.used_links[link_key] + self.bandwidth_request

            if total_used > current_bandwidth:
                return False
        return True

    def _deduct_path_bandwidth(self, new_links):
        """扣除路径带宽（带容错）"""
        for u, v in new_links:
            if self.graph.has_edge(u, v):
                link_key = tuple(sorted((u, v)))
                self.used_links[link_key] += self.bandwidth_request
                current_bw = self.graph[u][v].get('bandwidth', 1.0)
                self.graph[u][v]['bandwidth'] = max(current_bw - self.bandwidth_request, 0.0)

    def _add_path_to_tree(self, path):
        """添加路径到多播树"""
        new_links = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if not self.route_graph.has_edge(u, v):
                self.route_graph.add_edge(u, v)
                new_links.append((u, v))
        return new_links

    # ----------------------------------------------------------
    # 奖励函数（容错增强版）
    # ----------------------------------------------------------
    def _calculate_sfc_reward(self, path, node, cpu_req):
        """计算奖励（带完整容错）"""
        try:
            if len(self.route_graph.edges) == 0:
                global_qos = 0
            else:
                # 使用关键路径指标
                bw_vals, delay_vals, loss_vals = [], [], []

                for tree_node in self.tree_nodes:
                    if tree_node == self.start:
                        continue
                    try:
                        tree_path = nx.shortest_path(self.route_graph, self.start, tree_node)
                        path_delay = sum(self.graph[u][v].get('delay', 10.0)
                                         for u, v in zip(tree_path[:-1], tree_path[1:]))
                        path_bw = min(self.graph[u][v].get('bandwidth', 1.0)
                                      for u, v in zip(tree_path[:-1], tree_path[1:]))
                        path_loss = 1 - np.prod([1 - self.graph[u][v].get('loss', 0.01)
                                                 for u, v in zip(tree_path[:-1], tree_path[1:])])

                        delay_vals.append(path_delay)
                        bw_vals.append(path_bw)
                        loss_vals.append(path_loss)
                    except:
                        continue

                if not bw_vals:
                    # 回退到边指标
                    for (u, v) in self.route_graph.edges:
                        try:
                            idx_u = self.node_to_index.get(u, 0)
                            idx_v = self.node_to_index.get(v, 0)
                            if idx_u < len(self.nodes) and idx_v < len(self.nodes):
                                bw_vals.append(self.normal_bw_matrix[idx_u][idx_v])
                                delay_vals.append(self.normal_delay_matrix[idx_u][idx_v])
                                loss_vals.append(self.normal_loss_matrix[idx_u][idx_v])
                        except:
                            continue

                critical_delay = max(delay_vals) if delay_vals else 50.0
                critical_bw = min(bw_vals) if bw_vals else 0.5
                avg_loss = np.mean(loss_vals) if loss_vals else 0.05

                normalized_bw = np.clip(critical_bw / self.max_bandwidth, 0, 1)
                normalized_delay = np.clip(1 - (critical_delay / self.max_delay), 0, 1)
                normalized_loss = np.clip(1 - avg_loss, 0, 1)

                global_qos = (self.beta1 * normalized_bw +
                              self.beta2 * normalized_delay +
                              self.beta3 * normalized_loss)

            # QoS变化项
            delta_qos = global_qos - self.prev_global_qos
            delta_qos_term = (0.3 * delta_qos if delta_qos >= 0
                              else self.delta_qos_penalty * delta_qos)

            # 资源项
            cpu_total = self.graph.nodes[node].get('cpu_total', 1.0)
            cpu_ratio = np.clip(self.graph.nodes[node].get('cpu', 0.0) / max(cpu_total, 1e-6), 0, 1)
            resource_penalty = self.beta5 * (cpu_req / max(cpu_total, 1e-6))

            # 路径惩罚
            path_penalty = 0.1 * len(path) if path else 0
            stability_bonus = np.exp(-path_penalty) * 0.1

            reward = (global_qos + delta_qos_term + self.beta4 * cpu_ratio -
                      resource_penalty - path_penalty + stability_bonus)

            self.prev_global_qos = global_qos
            return float(reward)

        except Exception as e:
            self.logger.error(f"奖励计算错误: {e}")
            return 0.0

    # ----------------------------------------------------------
    # 辅助工具（容错增强版）
    # ----------------------------------------------------------
    def _has_valid_actions(self):
        """检查是否有有效动作"""
        try:
            return any(self.validate_action(n)[0] for n in self.nodes)
        except:
            return False

    def validate_action(self, node):
        """验证动作有效性（增强版）"""
        if node not in self.graph.nodes:
            return False, "节点不存在"

        if self.current_vnf_index >= len(self.vnfs):
            return False, "所有VNF已部署"

        cpu_req = self.vnfs[self.current_vnf_index]['cpu']
        node_cpu = self.graph.nodes[node].get('cpu', 0.0)

        if node_cpu < cpu_req:
            return False, "CPU资源不足"

        if node not in self.tree_nodes:
            # 检查连通性
            connected = False
            for tree_node in self.tree_nodes:
                try:
                    if nx.has_path(self.graph, tree_node, node):
                        # 额外检查路径带宽
                        test_path = nx.shortest_path(self.graph, tree_node, node)
                        if self._check_path_bandwidth(test_path):
                            connected = True
                            break
                except:
                    continue
            if not connected:
                return False, "节点不可达或带宽不足"

        return True, "有效动作"

    def get_valid_actions(self):
        """获取有效动作列表"""
        valid = []
        for node in self.nodes:
            try:
                ok, _ = self.validate_action(node)
                if ok:
                    valid.append(self.node_to_index[node])
            except:
                continue
        return valid

    def get_action_mask(self):
        """返回动作掩码"""
        valid_actions = self.get_valid_actions()
        mask = torch.zeros(len(self.action_space), dtype=torch.bool)
        for action in valid_actions:
            if 0 <= action < len(mask):
                mask[action] = True
        return mask

    # ----------------------------------------------------------
    # 新增方法：PKL文件读取和QoS参数获取
    # ----------------------------------------------------------
    def read_pickle_and_modify(self, pkl_path):
        """
        从pickle文件读取并修改图的权重（模拟动态网络状态）

        Args:
            pkl_path: pickle文件路径
        """
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)

            # 假设pkl文件包含边的权重修改信息
            if isinstance(data, dict):
                for (u, v), weights in data.items():
                    if self.graph.has_edge(u, v):
                        # 更新边属性
                        if 'bandwidth' in weights:
                            self.graph[u][v]['bandwidth'] = weights['bandwidth']
                        if 'delay' in weights:
                            self.graph[u][v]['delay'] = weights['delay']
                        if 'loss' in weights:
                            self.graph[u][v]['loss'] = weights['loss']

                # 重新初始化归一化矩阵
                self._init_graph_parameters()
                self.logger.info(f"成功加载并应用 {pkl_path} 的网络权重")

        except Exception as e:
            self.logger.warning(f"加载pickle文件失败 {pkl_path}: {e}，使用默认权重")

    def get_sfc_qos_params(self):
        """
        获取当前SFC部署的QoS参数

        Returns:
            tuple: (平均剩余带宽, 最大延迟, 平均丢包率, 树边数)
        """
        try:
            if len(self.route_graph.edges) == 0:
                return 0.0, 0.0, 0.0, 0

            # 计算所有部署路径的QoS指标
            total_bw = []
            total_delay = []
            total_loss = []

            for u, v in self.route_graph.edges:
                if self.graph.has_edge(u, v):
                    total_bw.append(self.graph[u][v].get('bandwidth', 0.0))
                    total_delay.append(self.graph[u][v].get('delay', 0.0))
                    total_loss.append(self.graph[u][v].get('loss', 0.0))

            # 计算关键路径指标
            avg_bw = np.mean(total_bw) if total_bw else 0.0
            max_delay = np.max(total_delay) if total_delay else 0.0
            avg_loss = np.mean(total_loss) if total_loss else 0.0
            tree_length = len(self.route_graph.edges)

            return float(avg_bw), float(max_delay), float(avg_loss), int(tree_length)

        except Exception as e:
            self.logger.error(f"获取SFC QoS参数失败: {e}")
            return 0.0, 0.0, 0.0, 0

    def get_env_info(self):
        """获取环境信息（容错版）"""
        try:
            valid_actions = self.get_valid_actions()
            resource_utilization = np.mean([
                np.clip(1 - (self.graph.nodes[n].get('cpu', 0.0) /
                             max(self.graph.nodes[n].get('cpu_total', 1.0), 1e-6)), 0, 1)
                for n in self.nodes
            ])

            return {
                "nodes": len(self.nodes),
                "edges": len(self.edges),
                "vnfs": len(self.vnfs),
                "current_vnf_index": self.current_vnf_index,
                "tree_nodes": len(self.tree_nodes),
                "start": self.start,
                "done": self.done,
                "valid_actions_count": len(valid_actions),
                "resource_utilization": float(resource_utilization),
                "deployment_progress": self.current_vnf_index / max(len(self.vnfs), 1),
                "bandwidth_request": self.bandwidth_request,
                "multi_path_enabled": self.enable_multi_path_selection,
            }
        except Exception as e:
            self.logger.error(f"获取环境信息错误: {e}")
            return {"error": str(e)}

    def render(self, mode='human'):
        """环境渲染（容错版）"""
        if mode == 'human':
            try:
                progress = f"{self.current_vnf_index}/{len(self.vnfs)}" if self.vnfs else "0/0"
                vnf_req = self.vnfs[self.current_vnf_index]['cpu'] if not self.done and self.current_vnf_index < len(
                    self.vnfs) else '完成'
                print(f"SFC部署进度: {progress}")
                print(f"多播树节点数: {len(self.tree_nodes)}")
                print(f"有效动作数: {len(self.get_valid_actions())}")
                print(f"当前VNF需求: {vnf_req}")
                print(f"带宽请求: {self.bandwidth_request}")
                print(f"多路径选择: {'启用' if self.enable_multi_path_selection else '禁用'}")
            except Exception as e:
                print(f"渲染错误: {e}")

    def close(self):
        """环境清理"""
        self.logger.info("关闭环境")
        self.graph = None
        self.route_graph = None

    def enable_multi_path_mode(self, enabled=False, max_paths=3):
        """动态切换多路径选择模式"""
        self.enable_multi_path_selection = enabled
        self.max_candidate_paths = max_paths
        status = "启用" if enabled else "禁用"
        self.logger.info(f"多路径选择模式: {status} (最大路径数: {max_paths})")


# 单元测试函数
def test_sfc_env():
    """SFC环境单元测试"""
    import unittest

    class TestSFCEnv(unittest.TestCase):
        def setUp(self):
            # 创建测试图
            self.graph = nx.Graph()
            self.graph.add_nodes_from([1, 2, 3])
            self.graph.add_edges_from([(1, 2), (2, 3)])

            # 设置节点属性
            for node in self.graph.nodes():
                self.graph.nodes[node]['cpu'] = 10.0
                self.graph.nodes[node]['cpu_total'] = 20.0

            # 设置边属性
            for u, v in self.graph.edges():
                self.graph[u][v]['bandwidth'] = 100.0
                self.graph[u][v]['delay'] = 10.0
                self.graph[u][v]['loss'] = 0.01

            # 创建VNF链
            self.vnfs = [{'cpu': 2.0}, {'cpu': 3.0}]

            # 创建配置
            class Conf:
                bandwidth_request = 1.0
                path_cutoff = 4

            self.conf = Conf()

        def test_env_initialization(self):
            """测试环境初始化"""
            env = SFCEnv(self.graph, self.vnfs, self.conf)
            self.assertIsNotNone(env)
            self.assertEqual(len(env.action_space), 3)

        def test_reset(self):
            """测试环境重置"""
            env = SFCEnv(self.graph, self.vnfs, self.conf)
            state, info = env.reset()
            self.assertIsNotNone(state)
            self.assertIsInstance(state, torch.Tensor)

        def test_step(self):
            """测试动作执行"""
            env = SFCEnv(self.graph, self.vnfs, self.conf)
            env.reset()

            # 测试有效动作
            valid_actions = env.get_valid_actions()
            if valid_actions:
                state, reward, done, info = env.step(valid_actions[0])
                self.assertIsNotNone(state)
                self.assertIsInstance(reward, float)

        def test_invalid_action(self):
            """测试无效动作"""
            env = SFCEnv(self.graph, self.vnfs, self.conf)
            env.reset()

            # 测试越界动作
            state, reward, done, info = env.step(100)
            self.assertLess(reward, 0)

    # 运行测试
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSFCEnv)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    # 运行测试
    test_sfc_env()