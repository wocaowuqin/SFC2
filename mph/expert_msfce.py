# (文件: expert_msfce.py)
import numpy as np
from msfce_simulator import MSFCE_Solver  # 导入您的专家


class ExpertMSFCE(MSFCE_Solver):
    """
    一个“可观察的”专家，它在执行时会记录其内部决策。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trajectory_log = []

    def _get_flat_state(self, network_state: Dict, request: Dict):
        """将网络状态和请求展平为单个向量 (用于神经网络)"""
        # ... (这里需要您定义如何将状态字典转换为一个长向量)
        # 例如: np.concatenate([
        #     network_state['cpu'] / CPU_CAPACITY,
        #     network_state['mem'] / MEM_CAPACITY,
        #     network_state['bw'] / BW_CAPACITY,
        #     request['bw_origin'] / BW_CAPACITY,
        #     ...
        # ])
        pass

    def solve_and_log(self, request: Dict, network_state: Dict):
        """
        运行 MSFC-CE 算法，但将决策过程记录为 (State, Goal, Action) 轨迹。
        """
        self.trajectory_log.clear()

        # ... (MSFCE_Solver.solve_request 的大部分逻辑) ...

        # 1. 记录高层决策 (选择树干)
        # (在 solve_request 中找到 best_d_idx 的逻辑)
        flat_state = self._get_flat_state(network_state, request)
        expert_goal = best_d_idx  # 高层目标 (0-N dest)

        # 2. 记录低层决策 (如何连接树干)
        # (在 _calc_eval 中找到 best_k 的逻辑)
        # 我们需要将 (k) 映射为一个整数动作ID
        expert_action = k  # 低层动作 (0-K paths)

        self.trajectory_log.append((flat_state, expert_goal, expert_action, ...))

        # 3. 记录后续的 (高层, 低层) 决策
        while unadded:
            # (在 solve_request 循环中)
            flat_state = self._get_flat_state(network_state, request)
            expert_goal = best_d  # 高层目标 (下一个 d_idx)

            # (在 _calc_atnp 中找到 best_i_idx 和 best_k 的逻辑)
            # 我们需要将 (i_idx, k) 映射为一个整数动作ID
            # 假设 K=5, max_nodes=20
            # expert_action = best_i_idx * 5 + best_k
            expert_action = ...  # 低层动作

            self.trajectory_log.append((flat_state, expert_goal, expert_action, ...))

            # ... (应用决策, 更新状态)

        return self.trajectory_log