from collections import namedtuple
from torch import Tensor
import numpy as np
import torch
import math


class PPO:
    def __init__(self, conf):
        self.conf = conf
        self.memory = Memory()
        torch.manual_seed(self.conf.seed)

    def _normal_logproba(self, x, mean, logstd, std=None):
        if std is None:
            std = torch.exp(logstd)

        std_sq = std.pow(2)
        logproba = (
            -0.5 * math.log(2 * math.pi) - logstd - (x - mean).pow(2) / (2 * std_sq)
        )
        return logproba.sum(1)

    def select_action(self, action_mean, action_logstd, return_logproba=True):
        """
        given mean and std, sample an action from normal(mean, std)
        also returns probability of the given chosen
        """
        action_std = torch.exp(action_logstd)
        action = torch.normal(action_mean, action_std)
        if return_logproba:
            logproba = self._normal_logproba(
                action, action_mean, action_logstd, action_std
            )
        return action, logproba

    def process_traj(self):
        batch = self.memory.sample()
        self.batch_size = len(self.memory)

        # step2: extract variables from trajectories
        rewards = Tensor(batch.reward)
        values = Tensor(batch.value)
        masks = Tensor(batch.mask)
        self.actions = Tensor(batch.action)
        self.states = Tensor(batch.state)
        self.oldlogproba = Tensor(batch.logproba)

        deltas = Tensor(self.batch_size)
        self.returns = Tensor(self.batch_size)
        self.advantages = Tensor(self.batch_size)

        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(self.batch_size)):
            self.returns[i] = rewards[i] + self.conf.gamma * prev_return * masks[i]
            deltas[i] = rewards[i] + self.conf.gamma * prev_value * masks[i] - values[i]
            # ref: https://arxiv.org/pdf/1506.02438.pdf (generalization advantage estimate)
            self.advantages[i] = (
                deltas[i]
                + self.conf.gamma * self.conf.lamda * prev_advantage * masks[i]
            )

            prev_return = self.returns[i]
            prev_value = values[i]
            prev_advantage = self.advantages[i]
        if self.conf.advantage_norm:
            self.advantages = (self.advantages - self.advantages.mean()) / (
                self.advantages.std() + self.conf.EPS
            )

        # clear the memory
        self.memory = Memory()

    def get_sample(self):
        minibatch_ind = np.random.choice(
            self.batch_size, self.conf.minibatch_size, replace=False
        )
        minibatch_states = self.states[minibatch_ind]
        minibatch_actions = self.actions[minibatch_ind]
        minibatch_oldlogproba = self.oldlogproba[minibatch_ind]
        minibatch_advantages = self.advantages[minibatch_ind]
        minibatch_returns = self.returns[minibatch_ind]
        msg = (
            minibatch_states,
            minibatch_actions,
            minibatch_oldlogproba,
            minibatch_advantages,
            minibatch_returns,
        )
        return msg


Transition = namedtuple(
    "Transition",
    ("state", "value", "action", "logproba", "mask", "next_state", "reward"),
)


class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update:
            self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape


class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)
