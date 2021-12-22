import random
import math

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F

from msagent.utils.atari_wrappers import make_atari, wrap_Framestack
from model import Dueling_DQN, MLP

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

epsilon_max = 1
epsilon_min = 0.01
eps_decay = 30000


class Memory_Buffer(object):
    def __init__(self, memory_size=100000):
        self.buffer = []
        self.memory_size = memory_size
        self.next_idx = 0

    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        if len(self.buffer) <= self.memory_size:  # buffer not full
            self.buffer.append(data)
        else:  # buffer is full
            self.buffer[self.next_idx] = data
        self.next_idx = (self.next_idx + 1) % self.memory_size

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in range(batch_size):

            idx = random.randint(0, self.size() - 1)
            data = self.buffer[idx]
            state, action, reward, next_state, done = data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return torch.cat(states), actions, rewards, torch.cat(next_states), dones

    def size(self):
        return len(self.buffer)


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class Memory_Buffer_PER(object):
    # stored as ( s, a, r, s_ ) in SumTree
    def __init__(self, memory_size=1000, a=0.6, e=0.01):
        self.tree = SumTree(memory_size)
        self.memory_size = memory_size
        self.prio_max = 0.1
        self.a = a
        self.e = e

    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        p = (np.abs(self.prio_max) + self.e) ** self.a  # proportional priority
        self.tree.add(p, data)

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)

            state, action, reward, next_state, done = data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            priorities.append(p)
            idxs.append(idx)
        return idxs, torch.cat(states), actions, rewards, torch.cat(next_states), dones

    def update(self, idxs, errors):
        self.prio_max = max(self.prio_max, max(np.abs(errors)))
        for i, idx in enumerate(idxs):
            p = (np.abs(errors[i]) + self.e) ** self.a
            self.tree.update(idx, p)

    def size(self):
        return self.tree.n_entries


class DQN_Agent:
    def __init__(self, input_shape, action_space=None, USE_CUDA=False, memory_size=10000, prio_a=0.6, prio_e=0.001, epsilon=1, lr=1e-4, priority=True, mlp=False):
        self.epsilon = epsilon
        self.priority = priority
        self.mlp = mlp
        self.action_space = action_space
        if self.priority:
            self.memory_buffer = Memory_Buffer_PER(
                memory_size, a=prio_a, e=prio_e)
        else:
            self.memory_buffer = Memory_Buffer(memory_size)

        self.DQN = Dueling_DQN(input_shape=input_shape,
                               num_actions=action_space.n)
        self.DQN_target = Dueling_DQN(input_shape=input_shape,
                                      num_actions=action_space.n)
        if mlp:
            self.DQN = MLP(input_shape=input_shape, num_actions=action_space.n)
            self.DQN_target = MLP(input_shape=input_shape,
                                  num_actions=action_space.n)
        self.DQN_target.load_state_dict(self.DQN.state_dict())

        self.USE_CUDA = USE_CUDA
        if self.USE_CUDA:
            self.DQN = self.DQN.cuda()
            self.DQN_target = self.DQN_target.cuda()
        self.optimizer = optim.RMSprop(
            self.DQN.parameters(), lr=lr, eps=0.001, alpha=0.95)

    def observe(self, lazyframe):
        # from Lazy frame to tensor
        if self.mlp:
            return torch.unsqueeze(torch.FloatTensor(lazyframe), 0).cuda()
        state = torch.from_numpy(
            lazyframe._force().transpose(2, 0, 1)[None]/255).float()
        if self.USE_CUDA:
            state = state.cuda()
        return state

    def value(self, state):
        q_values = self.DQN(state)
        return q_values

    def act(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon

        q_values = self.value(state).cpu().detach().numpy()
        if random.random() < epsilon:
            aciton = random.randrange(self.action_space.n)
        else:
            aciton = q_values.argmax(1)[0]
        return aciton

    def compute_td_loss(self, idxs, states, actions, rewards, next_states, is_done, gamma=0.99):
        actions = torch.tensor(actions).long()    # shape: [batch_size]
        # shape: [batch_size]
        rewards = torch.tensor(rewards, dtype=torch.float)
        is_done = torch.tensor(is_done).bool()  # shape: [batch_size]

        if self.USE_CUDA:
            actions = actions.cuda()
            rewards = rewards.cuda()
            is_done = is_done.cuda()

        # get q-values for all actions in current states
        predicted_qvalues = self.DQN(states)

        # select q-values for chosen actions
        predicted_qvalues_for_actions = predicted_qvalues[
            range(states.shape[0]), actions
        ]

        # compute q-values for all actions in next states

        predicted_next_qvalues = self.DQN_target(next_states)

        # compute V*(next_states) using predicted next q-values
        next_state_values = predicted_next_qvalues.max(-1)[0]

        # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
        target_qvalues_for_actions = rewards + gamma * next_state_values  # YOUR CODE

        # at the last state we shall use simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
        target_qvalues_for_actions = torch.where(
            is_done, rewards, target_qvalues_for_actions)

        # mean squared error loss to minimize
        if self.priority:
            errors = (predicted_qvalues_for_actions -
                      target_qvalues_for_actions).detach().cpu().squeeze().tolist()
            self.memory_buffer.update(idxs, errors)
        loss = F.smooth_l1_loss(
            predicted_qvalues_for_actions, target_qvalues_for_actions.detach())

        return loss

    def learn_from_experience(self, batch_size):
        if self.memory_buffer.size() > batch_size:
            if self.priority:
                idxs, states, actions, rewards, next_states, dones = self.memory_buffer.sample(
                    batch_size)
            else:
                states, actions, rewards, next_states, dones = self.memory_buffer.sample(
                    batch_size)
                idxs = []
            td_loss = self.compute_td_loss(
                idxs, states, actions, rewards, next_states, dones)
            self.optimizer.zero_grad()
            td_loss.backward()
            for param in self.DQN.parameters():
                param.grad.data.clamp_(-1, 1)

            self.optimizer.step()
            return(td_loss.item())
        else:
            return(0)


# e-greedy decay
def epsilon_by_frame(frame_idx): return epsilon_min + (epsilon_max - epsilon_min) * math.exp(
    -1. * frame_idx / eps_decay)
