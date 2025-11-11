# (文件: hirl_sfc_agent.py)
from hyperparameters import *
from replay_buffer import PrioritizedReplayBuffer  # 可以重用
from schedules import LinearSchedule  # 可以重用


class Agent_SFC:
    """SFC 版本的 Agent 类"""

    def __init__(self, net: Hdqn_SFC, ...):
        self.net = net  # Hdqn_SFC 实例
        self.memory = PrioritizedReplayBuffer(...)
        self.exploration = LinearSchedule(...)
        self.controllerEpsilon = 1.0
        self.randomPlay = True  # 训练初期的标志
        # ...

    def selectMove(self, state_vector, goal_one_hot):
        """低层动作选择 (Epsilon-Greedy)"""
        if self.controllerEpsilon < random.random():
            q_values = self.net.controllerNet.predict([state_vector, goal_one_hot])
            return np.argmax(q_values)
        return random.choice(range(NB_ACTIONS))

    def criticize(self, sub_task_completed: bool, cost: float):
        """
        计算内部奖励 (Intrinsic Reward)
        我们希望“低成本”完成“高层指定”的子任务
        """
        reward = 0.0
        if sub_task_completed:
            reward += 1.0  # 完成任务

        # 奖励与成本负相关
        reward -= cost / MAX_EXPECTED_COST
        return reward

    def store(self, experience):
        """存储 (s, g, a, r, s', done)"""
        self.memory.add(experience.state, experience.goal, experience.action,
                        experience.reward, experience.next_state, experience.done)

    def update(self, stepCount):
        """
        DQN 更新。
        这与 atari 版本中的 _update 函数几乎相同。
        """
        # ... (从 memory 采样, 计算 target_q, train_on_batch)
        pass

    def store_imitation_data(self, expert_trajectory):
        """
        (新增) 模仿学习阶段: 将专家数据直接存入回放池
        """
        for (state, goal, action, next_state, done) in expert_trajectory:
            # 给予一个高的 "伪奖励" (1.0) 来激励模仿
            reward = 1.0
            exp = (state, goal, action, reward, next_state, done)
            self.store(exp)