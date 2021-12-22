import gym
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x


def to_torch(x, dtype=None, device='cpu'):
    if isinstance(x, np.ndarray) and issubclass(
            x.dtype.type, (np.bool_, np.number)
    ):
        x = torch.from_numpy(x).to(device)
        if dtype is not None:
            x = x.type(dtype)
        return x


def to_torch_as(x, y):
    assert isinstance(y, torch.Tensor)
    return to_torch(x, dtype=y.dtype, device=y.device)


def miniblock(input_size, output_size,
              norm_layer=None, activation=None):
    layers = [nn.Linear(input_size, output_size)]
    if norm_layer is not None:
        layers += [norm_layer(output_size)]
    if activation is not None:
        layers += [activation()]
    return layers


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=(),
                 norm_layer=None, activation=nn.ReLU, device=None):
        super().__init__()
        self.device = device
        if norm_layer:
            if isinstance(norm_layer, list):
                assert len(norm_layer) == len(hidden_sizes)
                norm_layer_list = norm_layer
            else:
                norm_layer_list = [
                    norm_layer for _ in range(len(hidden_sizes))
                ]
        else:
            norm_layer_list = [None] * len(hidden_sizes)
        if activation:
            if isinstance(activation, list):
                assert len(activation) == len(hidden_sizes)
                activation_list = activation
            else:
                activation_list = [
                    activation for _ in range(len(hidden_sizes))
                ]
        else:
            activation_list = [None] * len(hidden_sizes)
        hidden_sizes = [input_dim] + list(hidden_sizes)
        model = []
        for in_dim, out_dim, norm, activ in zip(
                hidden_sizes[:-1], hidden_sizes[1:],
                norm_layer_list, activation_list
        ):
            model += miniblock(in_dim, out_dim, norm, activ)
        if output_dim > 0:
            model += [nn.Linear(hidden_sizes[-1], output_dim)]
        self.output_dim = output_dim | hidden_sizes[-1]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = torch.as_tensor(
            x, device=self.device, dtype=torch.float32
        )
        return self.model(x.flatten(1))


class Net(nn.Module):
    def __init__(self, state_shape, action_shape=0, hidden_sizes=(),
                 norm_layer=None, activation=nn.ReLU, device='cpu',
                 softmax=False, concat=False):
        super().__init__()
        self.device = device
        self.softmax = softmax
        input_dim = int(np.prod(state_shape))
        action_dim = int(np.prod(action_shape))
        if concat:
            input_dim += action_dim
        output_dim = action_dim if not concat else 0
        self.model = MLP(input_dim, output_dim, hidden_sizes,
                         norm_layer, activation, device)
        self.output_dim = self.model.output_dim

    def forward(self, s):
        logits = self.model(s)
        # bsz = logits.shape[0]
        if self.softmax:
            logits = torch.softmax(logits, dim=-1)
        return logits


class Actor(nn.Module):
    def __init__(self, preprocess_net, action_shape, hidden_sizes=(),
                 max_action=1.0, device='cpu',
                 preprocess_net_output_dim=None):
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = int(np.prod(action_shape))
        input_dim = getattr(preprocess_net, "output_dim",
                            preprocess_net_output_dim)
        self.last = MLP(input_dim, self.output_dim,
                        hidden_sizes, device=self.device)
        self._max = max_action

    def forward(self, s):
        logits = self.preprocess(s)
        logits = self._max * torch.tanh(self.last(logits))
        return logits


class Critic(nn.Module):
    def __init__(self, preprocess_net, hidden_sizes=(),
                 device='cpu', preprocess_net_output_dim=None):
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = 1
        input_dim = getattr(preprocess_net, "output_dim",
                            preprocess_net_output_dim)
        self.last = MLP(input_dim, 1, hidden_sizes, device=self.device)

    def forward(self, s, a):
        s = torch.as_tensor(s, device=self.device,
                            dtype=torch.float32).flatten(1)
        if a is not None:
            a = torch.as_tensor(a, device=self.device,
                                dtype=torch.float32).flatten(1)
            s = torch.cat([s, a], dim=1)
        logits = self.preprocess(s)
        logits = self.last(logits)
        return logits


class DDPG(nn.Module):
    def __init__(self, actor, actor_optim, critic, critic_optim,
                 tau=0.005, gamma=0.99, exploration_noise=None,
                 reward_normalization=False, action_scaling=True,
                 action_bound_method='clip', action_space=None):
        super().__init__()
        if actor is not None and actor_optim is not None:
            self.actor, self.actor_optim = actor, actor_optim
            self.actor_old = deepcopy(actor)
            self.actor_old.eval()
        if critic is not None and critic_optim is not None:
            self.critic, self.critic_optim = critic, critic_optim
            self.critic_old = deepcopy(critic)
            self.critic_old.eval()
        assert 0.0 <= tau <= 1.0, "tau should be in [0, 1]"
        self._tau = tau
        assert 0.0 <= gamma <= 1.0, "gamma should be in [0, 1]"
        self._gamma = gamma
        self._noise = exploration_noise
        self._rew_norm = reward_normalization
        # self._n_step = estimation_step

        self.action_scaling = action_scaling
        assert action_bound_method in ("", "clip", "tanh")
        self.action_bound_method = action_bound_method

        self.action_space = action_space

    def set_exp_noise(self, noise):
        self._noise = noise

    def train(self, mode=True):
        self.training = mode
        self.actor.train(mode)
        self.critic.train(mode)
        return self

    def sync_weight(self):
        for o, n in zip(self.actor_old.parameters(), self.actor.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic_old.parameters(),
                        self.critic.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

    def _target_q(self, batch):
        target_q = self.critic_old(
            batch["obs_next"],
            self(batch, model='actor_old', input='obs_next')
        )
        return target_q

    def process_fn(self, batch, buffer, indice):
        batch = self.compute_1step_return(batch, gamma=self._gamma)
        return batch

    def post_process_fn(self, batch, buffer, indice):
        pass

    def forward(self, batch, model="actor", input="obs"):
        model = getattr(self, model)
        obs = batch[input]
        act = model(obs)
        return act

    def map_action(self, act):
        if isinstance(self.action_space, gym.spaces.Box) and \
                isinstance(act, np.ndarray):
            if self.action_bound_method == "clip":
                act = np.clip(act, -1.0, 1.0)
            if self.action_scaling:
                assert np.min(act) >= -1.0 and np.max(act) <= 1.0, \
                    "action scaling only accepts raw action range = [-1, 1]"
                low, high = self.action_space.low, self.action_space.high
                act = low + (high - low) * (act + 1.0) / 2.0
        return act

    @staticmethod
    def value_mask(buffer, indice):
        mask = ~buffer.done[indice]
        return mask

    def compute_1step_return(self, batch, gamma=0.99):
        rew = batch["rew"].reshape(-1, 1)
        with torch.no_grad():
            target_q_torch = self._target_q(batch)
        target_q = to_numpy(target_q_torch)
        q_mask = ~batch["done"].reshape(-1, 1)
        target_q = target_q * q_mask
        target = rew + gamma * target_q

        batch.update(returns=to_torch_as(target, target_q_torch))
        return batch

    def update(self, sample_size, buffer):
        batch, indice = buffer.sample(sample_size)
        self.updating = True
        batch = self.process_fn(batch, buffer, indice)
        result = self.learn(batch)
        self.post_process_fn(batch, buffer, indice)
        self.updating = False
        return result

    def learn(self, batch):
        td, critic_loss = self._mse_optimizer(
            batch, self.critic, self.critic_optim
        )
        batch.update(weight=td)  # prio-buffer
        # actor
        action = self(batch)
        actor_loss = -self.critic(batch["obs"], action).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.sync_weight()
        return {
            "loss/actor": actor_loss.item(),
            "loss/critic": critic_loss.item(),
        }

    def exploration_noise(self, act):
        if self._noise is None:
            return act
        if isinstance(act, np.ndarray):
            noise = np.random.normal(0.0, self._noise, act.shape)
            return act + noise
        return act

    @staticmethod
    def _mse_optimizer(batch, critic, optimizer):
        weight = batch.get("weight", 1.0)
        current_q = critic(batch["obs"], batch["act"]).flatten()
        target_q = batch["returns"].flatten()
        td = current_q - target_q
        critic_loss = (td.pow(2) * weight).mean()
        optimizer.zero_grad()
        critic_loss.backward()
        optimizer.step()
        return td, critic_loss
