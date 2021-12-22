import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


class MLP(nn.Module):
    def __init__(self, input_shape, num_actions=5, hidden=50):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_shape[0], hidden)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(hidden, num_actions)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_value = self.out(x)
        return action_value


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions=5):
        super(DQN, self).__init__()
        self.input_shape = input_shape
        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.value = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        value = self.value(x)
        return value

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
