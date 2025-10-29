import gym
from gym import spaces
import numpy as np
import requests
import json
from typing import Dict, List, Tuple


class SFCEnvironment(gym.Env):
    """
    SFC部署环境 - 连接Ryu控制器
    """

    def __init__(self, ryu_url='http://127.0.0.1:8080',
                 num_nodes=12, max_vnfs_per_sfc=5):
        super().__init__()

        self.ryu_url = ryu_url
        self.num_nodes = num_nodes
        self.max_vnfs_per_sfc = max_vnfs_per_sfc

        # 状态空间：网络状态 + SFC需求
        # [node_cpu_1, ..., node_cpu_n, link_delay_1, ..., vnf_type, vnf_resources]
        self.state_dim = num_nodes * 3 + 10  # 节点状态 + 链路状态 + 当前VNF信息
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.state_dim,), dtype=np.float32
        )

        # 动作空间：选择部署节点
        self.action_space = spaces.Discrete(num_nodes)

        # 当前SFC请求
        self.current_sfc = None
        self.current_vnf_idx = 0
        self.deployed_vnfs = []

        # VNF类型定义
        self.vnf_types = {
            'firewall': {'cpu': 20, 'memory': 512},
            'ids': {'cpu': 30, 'memory': 1024},
            'nat': {'cpu': 15, 'memory': 256},
            'load_balancer': {'cpu': 25, 'memory': 512},
            'waf': {'cpu': 35, 'memory': 1024}
        }

    def reset(self) -> np.ndarray:
        """重置环境，开始新的episode"""
        # 调用Ryu API重置环境
        try:
            response = requests.post(f'{self.ryu_url}/sfc/reset')
            response.raise_for_status()
        except Exception as e:
            print(f"Warning: Failed to reset Ryu environment: {e}")

        # 生成新的SFC请求
        self.current_sfc = self._generate_sfc_request()
        self.current_vnf_idx = 0
        self.deployed_vnfs = []

        # 获取初始状态
        state = self._get_state()
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行动作：将当前VNF部署到指定节点
        """
        # 获取当前要部署的VNF信息
        vnf = self.current_sfc['vnfs'][self.current_vnf_idx]
        vnf_id = f"sfc_{self.current_sfc['id']}_vnf_{self.current_vnf_idx}"

        # 获取前一个VNF的位置（用于路径计算）
        prev_vnf_node = None
        if self.deployed_vnfs:
            prev_vnf_node = self.deployed_vnfs[-1]['node']
        elif 'source_node' in self.current_sfc:
            prev_vnf_node = self.current_sfc['source_node']

        # 调用Ryu API部署VNF
        try:
            deploy_data = {
                'vnf_id': vnf_id,
                'vnf_type': vnf['type'],
                'target_node': action,
                'resources': vnf['resources'],
                'prev_vnf_node': prev_vnf_node
            }

            response = requests.post(
                f'{self.ryu_url}/sfc/deploy',
                data=json.dumps(deploy_data),
                headers={'Content-Type': 'application/json'}
            )
            result = response.json()

            success = result['success']
            reward = result['reward']
            path = result.get('path', [])

        except Exception as e:
            print(f"Error deploying VNF: {e}")
            success = False
            reward = -10.0
            path = []

        # 记录部署
        if success:
            self.deployed_vnfs.append({
                'vnf_id': vnf_id,
                'type': vnf['type'],
                'node': action,
                'path': path
            })

        # 移动到下一个VNF
        self.current_vnf_idx += 1
        done = self.current_vnf_idx >= len(self.current_sfc['vnfs'])

        # 如果完成整个SFC链，给予额外奖励
        if done and success:
            reward += self._evaluate_sfc_quality()

        # 获取新状态
        next_state = self._get_state()

        info = {
            'deployed_vnfs': len(self.deployed_vnfs),
            'total_vnfs': len(self.current_sfc['vnfs']),
            'success': success
        }

        return next_state, reward, done, info

    def _get_state(self) -> np.ndarray:
        """获取当前状态向量"""
        try:
            # 从Ryu获取网络状态
            response = requests.get(f'{self.ryu_url}/sfc/network_state')
            network_state = response.json()

            # 构建状态向量
            state = np.zeros(self.state_dim, dtype=np.float32)

            # 节点状态 (CPU使用率)
            nodes = network_state['nodes']
            for i, node in enumerate(nodes[:self.num_nodes]):
                state[i] = node['cpu_usage'] / 100.0
                state[self.num_nodes + i] = len(node['vnfs']) / 10.0  # 归一化VNF数量
                state[2 * self.num_nodes + i] = node['available_cpu'] / 100.0

            # 当前VNF信息
            if self.current_vnf_idx < len(self.current_sfc['vnfs']):
                vnf = self.current_sfc['vnfs'][self.current_vnf_idx]
                vnf_type_idx = list(self.vnf_types.keys()).index(vnf['type'])
                state[-10] = vnf_type_idx / len(self.vnf_types)
                state[-9] = vnf['resources']['cpu'] / 100.0
                state[-8] = vnf['resources']['memory'] / 4096.0
                state[-7] = self.current_vnf_idx / len(self.current_sfc['vnfs'])

            return state

        except Exception as e:
            print(f"Error getting state: {e}")
            return np.zeros(self.state_dim, dtype=np.float32)

    def get_valid_actions(self) -> List[int]:
        """获取当前可用的动作（资源充足的节点）"""
        try:
            response = requests.get(f'{self.ryu_url}/sfc/network_state')
            network_state = response.json()

            vnf = self.current_sfc['vnfs'][self.current_vnf_idx]
            required_cpu = vnf['resources']['cpu']
            required_memory = vnf['resources']['memory']

            valid_nodes = []
            for node in network_state['nodes']:
                if (node['available_cpu'] >= required_cpu and
                        node.get('available_memory', 8192) >= required_memory):
                    valid_nodes.append(node['id'] - 1)  # 转换为0-based索引

            return valid_nodes if valid_nodes else list(range(self.num_nodes))

        except Exception as e:
            print(f"Error getting valid actions: {e}")
            return list(range(self.num_nodes))

    def predict_state(self, action: int) -> Dict:
        """预测部署到某节点后的状态"""
        try:
            vnf = self.current_sfc['vnfs'][self.current_vnf_idx]

            predict_data = {
                'vnf_id': f"temp_{self.current_vnf_idx}",
                'target_node': action,
                'resources': vnf['resources']
            }

            response = requests.post(
                f'{self.ryu_url}/sfc/predict_state',
                data=json.dumps(predict_data),
                headers={'Content-Type': 'application/json'}
            )
            prediction = response.json()
            return prediction

        except Exception as e:
            print(f"Error predicting state: {e}")
            return {}

    def _generate_sfc_request(self) -> Dict:
        """生成随机SFC请求"""
        num_vnfs = np.random.randint(2, self.max_vnfs_per_sfc + 1)

        vnf_chain = []
        for i in range(num_vnfs):
            vnf_type = np.random.choice(list(self.vnf_types.keys()))
            vnf_chain.append({
                'type': vnf_type,
                'resources': self.vnf_types[vnf_type].copy()
            })

        sfc = {
            'id': np.random.randint(1000, 9999),
            'vnfs': vnf_chain,
            'source_node': np.random.randint(0, self.num_nodes),
            'qos_requirements': {
                'max_delay': np.random.uniform(10, 50),  # ms
                'min_bandwidth': np.random.uniform(100, 1000)  # Mbps
            }
        }

        return sfc

    def _evaluate_sfc_quality(self) -> float:
        """评估整条SFC链的质量"""
        # 计算总路径长度
        total_hops = sum(len(vnf.get('path', [])) for vnf in self.deployed_vnfs)

        # 计算负载均衡度
        node_loads = {}
        for vnf in self.deployed_vnfs:
            node = vnf['node']
            node_loads[node] = node_loads.get(node, 0) + 1

        load_variance = np.var(list(node_loads.values())) if node_loads else 0

        # 综合评分
        quality_reward = 0
        quality_reward -= total_hops * 0.2  # 路径越短越好
        quality_reward -= load_variance * 0.5  # 负载越均衡越好
        quality_reward += len(self.deployed_vnfs) * 1.0  # 成功部署奖励

        return quality_reward

    def render(self, mode='human'):
        """可视化当前状态"""
        print(f"\n=== SFC Deployment Status ===")
        print(f"SFC ID: {self.current_sfc['id']}")
        print(f"Progress: {self.current_vnf_idx}/{len(self.current_sfc['vnfs'])}")
        print(f"Deployed VNFs:")
        for vnf in self.deployed_vnfs:
            print(f"  - {vnf['vnf_id']} ({vnf['type']}) -> Node {vnf['node']}")
            if vnf['path']:
                print(f"    Path: {' -> '.join(map(str, vnf['path']))}")
        print("=" * 30)