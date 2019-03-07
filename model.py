import torch
import torch.nn as nn
import torch.nn.functional as f


class QNetwork(nn.Module):
    """
    Actor (Policy) Model.
    """

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """
        Initialize parameters and build model.

        :param state_size (int): Dimension of each state
        :param action_size (int): Dimension of each action
        :param seed (int): Random seed
        :param fc1_units (int): Number of nodes in first hidden layer
        :param fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """
        Build a network that maps state -> action values.

        :param state: state values
        :return action values
        """
        x = f.relu(self.fc1(state))
        x = f.relu(self.fc2(x))
        return self.fc3(x)
