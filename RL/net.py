# -*- coding: utf-8 -*-
# @File    : net.py
# @Date    : 2025-10-14
# @Author  :
# @Desc    : Network Architectures for SFC DRL Agent (DQN/Dueling DQN)

import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(m):
    """
    Xavier/Glorot Initialization for Conv2d and Linear layers.
    """
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        # Use xavier_normal_ for Conv2d weights
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        # Use xavier_normal_ for Linear weights
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.constant_(m.bias, 0.0)


class MyMulticastNet(nn.Module):
    def __init__(self, states_channel, action_num):
        """
        Standard DQN architecture with simple 1x1 Conv2d.
        Assumes input state is a C x N x N tensor where N is the number of nodes.
        Flattening implies N=14.
        """
        super(MyMulticastNet, self).__init__()
        self.conv1 = nn.Conv2d(states_channel, 32, kernel_size=(1, 1))
        self.relu1 = nn.ReLU()

        # Input to fc1: 32 channels * 14 * 14 = 6272
        self.fc1 = nn.Linear(32 * 14 * 14, 512)
        self.fc1_relu = nn.ReLU()

        # Dueling-like structure for the final layers (simplified)
        self.adv1 = nn.Linear(512, 256)
        self.adv_relu = nn.ReLU()
        self.adv2 = nn.Linear(256, action_num)

        self.apply(weight_init)

    def forward(self, x):
        x = self.relu1(self.conv1(x))

        # Flatten all dimensions except the batch size
        x = x.view(x.shape[0], -1)
        x = self.fc1_relu(self.fc1(x))

        adv = self.adv_relu(self.adv1(x))
        q_value = self.adv2(adv)

        return q_value


class MyMulticastNet2(nn.Module):
    def __init__(self, states_channel, action_num):
        """
        Dueling DQN Architecture (D-DQN).
        Assumes input state is N=14 (as 32*14*14=6272 is used for fc1 input).
        """
        super(MyMulticastNet2, self).__init__()
        self.conv1 = nn.Conv2d(states_channel, 32, kernel_size=(1, 1))
        self.relu1 = nn.ReLU()

        # The original value 6272 is correct for 32 * 14 * 14
        self.fc1 = nn.Linear(6272, 512)
        self.fc1_relu = nn.ReLU()

        self.fc2 = nn.Linear(512, 256)
        self.fc2_relu = nn.ReLU()

        # Advantage Stream
        self.adv = nn.Linear(256, action_num)
        # Value Stream
        self.val = nn.Linear(256, 1)

        self.apply(weight_init)

    def forward(self, x):
        x = self.relu1(self.conv1(x))

        x = x.view(x.shape[0], -1)
        x = self.fc1_relu(self.fc1(x))
        x = self.fc2_relu(self.fc2(x))

        sate_value = self.val(x)
        advantage_function = self.adv(x)

        # Combine streams: Q = V(s) + (A(s, a) - mean(A(s, a)))
        return sate_value + (advantage_function - advantage_function.mean(dim=1, keepdim=True))


class MyMulticastNet3(nn.Module):
    def __init__(self, states_channel, action_num):
        """
        Dueling DQN with Parallel 1D Convolutions (Row/Column feature extraction).
        """
        super(MyMulticastNet3, self).__init__()
        # Convolutions along the first spatial dimension (rows/node index 1)
        self.conv1_1 = nn.Conv2d(states_channel, 32, kernel_size=(5, 1))
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=(5, 1))

        # Convolutions along the second spatial dimension (columns/node index 2)
        self.conv2_1 = nn.Conv2d(states_channel, 32, kernel_size=(1, 5))
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=(1, 5))

        # Flattening size 5376 implies input size N=18 (32*(18-5+1)*(18-5+1) = 32*13*13 = 5408)
        # Or N=14 (32*(14-5+1)*(14-5+1) = 32*10*10 = 3200).
        # The original 5376 suggests a custom input size, but we keep the original value.
        self.fc1 = nn.Linear(5376, 512)
        self.fc2 = nn.Linear(512, 256)

        self.adv = nn.Linear(256, action_num)
        self.val = nn.Linear(256, 1)

        self.apply(weight_init)

    def forward(self, x):
        x1_1 = F.leaky_relu(self.conv1_1(x))
        x1_2 = F.leaky_relu(self.conv1_2(x1_1))

        x2_1 = F.leaky_relu(self.conv2_1(x))
        x2_2 = F.leaky_relu(self.conv2_2(x2_1))

        x1_3 = x1_2.view(x.shape[0], -1)
        x2_3 = x2_2.view(x.shape[0], -1)
        # Concatenate features from both streams
        x = torch.cat([x1_3, x2_3], dim=1)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))

        sate_value = self.val(x)
        advantage_function = self.adv(x)

        return sate_value + (advantage_function - advantage_function.mean(dim=1, keepdim=True))


class MyMulticastNet4(nn.Module):
    def __init__(self, states_channel, action_num):
        """
        CNN+LSTM for Sequential Decision DRL (e.g., A3C/Recurrent DQN).
        Requires hidden and cell states (hx, cx) as input.
        """
        super(MyMulticastNet4, self).__init__()
        # Parallel CNN Stream 1
        self.conv1_1 = nn.Conv2d(states_channel, 32, kernel_size=(3, 1))
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=(3, 1))
        self.conv1_3 = nn.Conv2d(32, 32, kernel_size=(3, 1))

        # Parallel CNN Stream 2
        self.conv2_1 = nn.Conv2d(states_channel, 32, kernel_size=(1, 3))
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=(1, 3))
        self.conv2_3 = nn.Conv2d(32, 32, kernel_size=(1, 3))

        # Feature size 3584 implies N=10 after the three 3x1 and 1x3 convs
        self.lstm = nn.LSTMCell(3584, 512)
        self.fc1 = nn.Linear(512, 256)

        self.adv = nn.Linear(256, action_num)
        self.val = nn.Linear(256, 1)

        self.apply(weight_init)
        # Initialize LSTM biases to zeros for better stability
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

    def forward(self, inputs):
        """
        :param inputs: Tuple (x, (hx, cx)) where x is the state and (hx, cx) are LSTM states.
        :return: (Q_value, (hx, cx))
        """
        x, (hx, cx) = inputs

        # Stream 1
        x1_1 = F.leaky_relu(self.conv1_1(x))
        x1_2 = F.leaky_relu(self.conv1_2(x1_1))
        x1_3 = F.leaky_relu(self.conv1_3(x1_2))

        # Stream 2
        x2_1 = F.leaky_relu(self.conv2_1(x))
        x2_2 = F.leaky_relu(self.conv2_2(x2_1))
        x2_3 = F.leaky_relu(self.conv2_3(x2_2))

        x1_3 = x1_3.view(x.shape[0], -1)
        x2_3 = x2_3.view(x.shape[0], -1)
        # Sum the features (used instead of cat in MyMulticastNet3)
        x = x1_3 + x2_3

        # Pass through LSTM
        hx, cx = self.lstm(x, (hx, cx))

        x = hx
        x = F.leaky_relu(self.fc1(x))

        sate_value = self.val(x)
        advantage_function = self.adv(x)

        # Return Q-value and the updated hidden/cell states
        q_value = sate_value + (advantage_function - advantage_function.mean(dim=1, keepdim=True))

        return q_value, (hx, cx)


# --------------------------------------------------------------------------
# Aliases for DRL framework compatibility (e.g., train_sfc.py expects 'Net')
# Dueling DQN is generally preferred over standard DQN.
# --------------------------------------------------------------------------
Net = MyMulticastNet2