# (文件: hirl_sfc_models.py)
from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate

# 假设:
STATE_VECTOR_SIZE = 150  # (例如: 28 CPU + 28 Mem + 40 BW + 54 Request info...)
NB_SUBTASKS = 10  # (最大目的地数量)
NB_ACTIONS = 100  # (最大低层动作空间, e.g., 20个连接点 * 5条K-path)


def create_meta_controller(state_shape=(STATE_VECTOR_SIZE,)):
    """
    高层元控制器 (SFC 版本的 MetaNN)
    输入: 状态
    输出: 哪个子任务 (d_idx)
    """
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=state_shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(NB_SUBTASKS, activation='softmax'))  # 输出选择子任务的概率
    # ... (编译)
    return model


def create_low_level_controller(state_shape=(STATE_VECTOR_SIZE,), goal_shape=(NB_SUBTASKS,)):
    """
    低层控制器 (SFC 版本的 Hdqn)
    输入: 状态 + 子任务
    输出: 动作的 Q-Value
    """
    state_input = Input(shape=state_shape, name='state_input')
    goal_input = Input(shape=goal_shape, name='goal_input')  # 独热编码的目标

    merged_input = concatenate([state_input, goal_input])

    x = Dense(256, activation='relu')(merged_input)
    x = Dense(256, activation='relu')(x)
    output = Dense(NB_ACTIONS, activation='linear')(x)  # 输出 Q-values

    model = Model(inputs=[state_input, goal_input], outputs=output)
    # ... (编译)
    return model


class Hdqn_SFC:
    """SFC 版本的 Hdqn 类, 封装 controller 和 target_controller"""

    def __init__(self):
        self.controllerNet = create_low_level_controller()
        self.targetControllerNet = create_low_level_controller()
        self.targetControllerNet.set_weights(self.controllerNet.get_weights())