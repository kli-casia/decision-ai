import torch
import numpy as np
from model import Net, Actor, Critic, DDPG
from model import to_numpy


class CircularArray:
    def __init__(self, maxsize=1):
        super().__init__()
        self._maxsize = maxsize
        self.reset()

    def __getitem__(self, index):
        return self.data[index]

    def reset(self):
        self.data = np.full(self._maxsize, None)
        self._size, self._ptr = 0, -1

    def append(self, item):
        self._ptr = (self._ptr + 1) % self._maxsize
        self.data[self._ptr] = item
        self._size = min(self._size + 1, self._maxsize)


class Buffer:
    _transition_keys = ['obs', 'act', 'rew', 'obs_next', 'done', 'info']

    def __init__(self, maxsize):
        self._maxsize = maxsize
        self.reset()

    def __getattr__(self, key):
        try:
            return self.data[key]
        except KeyError as e:
            raise AttributeError from e

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.data.items()}

    def reset(self):
        self.data = {k: CircularArray(self._maxsize)
                     for k in self._transition_keys}
        self._size, self._ptr = 0, -1

    def append(self, transition):
        self._ptr = (self._ptr + 1) % self._maxsize
        for k in self.data.keys():
            self.data[k].append(transition[k])
        self._size = min(self._size + 1, self._maxsize)

    def next(self, index):
        end_flag = self.done[index] | (index == self._ptr)
        return (index + (1 - end_flag)) % self._maxsize

    def sample(self, batch_size, replace=True):
        indices = np.random.choice(self._size, batch_size, replace=replace)
        batch = self[indices]
        for k, v in batch.items():
            batch[k] = np.stack(v)
        return batch, indices


def build_model(state_shape, action_shape, hidden_sizes,
                actor_lr, critic_lr, action_space, max_action, device,
                tau, gamma, exploration_noise):
    net_a = Net(state_shape, hidden_sizes=hidden_sizes, device=device)
    actor = Actor(
        net_a,
        action_shape,
        max_action=max_action,
        device=device).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    net_c = Net(
        state_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        concat=True,
        device=device)
    critic = Critic(net_c, device=device).to(device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    policy = DDPG(actor, actor_optim, critic, critic_optim, tau=tau, gamma=gamma,
                  exploration_noise=exploration_noise, action_space=action_space)
    return policy


def reset(env):
    _transition_keys = ['obs', 'act', 'rew', 'obs_next', 'done', 'info']
    transition = dict.fromkeys(_transition_keys)
    transition.update(obs=env.reset())
    return transition


def preprocess_buffer(buffer, env, n_step):
    transition = reset(env)
    step_count = 0
    while True:
        act = env.action_space.sample()
        obs_next, rew, done, info = env.step(act)
        transition.update(
            act=act,
            obs_next=obs_next,
            rew=rew,
            done=done,
            info=info)
        buffer.append(transition.copy())
        step_count += 1
        if done:
            transition.update(obs=env.reset())
        else:
            transition.update(obs=obs_next)

        if step_count >= n_step:
            break


def policy_eval(policy, obs):
    with torch.no_grad():
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
        act = policy.actor(obs).flatten()
    act = to_numpy(act)
    return act


def test_episode(env, policy, n_episode=1):
    transition = reset(env)
    episode_count = 0
    ep_rews, ep_rew = [], 0
    ep_lens, ep_len = [], 0
    while True:
        obs = transition['obs']
        act = policy_eval(policy, obs)
        obs_next, rew, done, info = env.step(act)
        transition.update(
            act=act,
            obs_next=obs_next,
            rew=rew,
            done=done,
            info=info)
        ep_len += 1
        ep_rew += rew
        if done:
            transition.update(obs=env.reset())
            ep_rews.append(ep_rew)
            ep_lens.append(ep_len)
            ep_rew, ep_len = 0, 0
            episode_count += 1
        else:
            transition.update(obs=obs_next)

        if episode_count >= n_episode:
            break
    test_result = {'rew': np.mean(ep_rews), 'rew_std': np.std(ep_rews),
                   'len': np.mean(ep_lens), 'len_std': np.std(ep_lens)}
    return test_result
