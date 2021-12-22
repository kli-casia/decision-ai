import torch.optim as opt
import torch.nn as nn
import torch
import math

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, layer_norm=True, lossvalue_norm=True):
        super(ActorCritic, self).__init__()
        self.lossvalue_norm = lossvalue_norm
        self.loss_coeff_value = 0.5
        self.loss_coeff_entropy = 0.01
        self.lr = 3e-4
        
        self.actor_fc1 = nn.Linear(num_inputs, 64)
        self.actor_fc2 = nn.Linear(64, 64)
        self.actor_fc3 = nn.Linear(64, num_outputs)
        self.actor_logstd = nn.Parameter(torch.zeros(1, num_outputs))

        self.critic_fc1 = nn.Linear(num_inputs, 64)
        self.critic_fc2 = nn.Linear(64, 64)
        self.critic_fc3 = nn.Linear(64, 1)

        if layer_norm:
            self.layer_norm(self.actor_fc1, std=1.0)
            self.layer_norm(self.actor_fc2, std=1.0)
            self.layer_norm(self.actor_fc3, std=0.01)

            self.layer_norm(self.critic_fc1, std=1.0)
            self.layer_norm(self.critic_fc2, std=1.0)
            self.layer_norm(self.critic_fc3, std=1.0)

        self.optimizer = opt.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)

    def forward(self, states):
        """
        run policy network (actor) as well as value network (critic)
        :param states: a Tensor2 represents states
        :return: 3 Tensor2
        """
        action_mean, action_logstd = self._forward_actor(states)
        critic_value = self._forward_critic(states)
        return action_mean, action_logstd, critic_value

    def _forward_actor(self, states):
        x = torch.tanh(self.actor_fc1(states))
        x = torch.tanh(self.actor_fc2(x))
        action_mean = self.actor_fc3(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        return action_mean, action_logstd

    def _forward_critic(self, states):
        x = torch.tanh(self.critic_fc1(states))
        x = torch.tanh(self.critic_fc2(x))
        critic_value = self.critic_fc3(x)
        return critic_value


    @staticmethod
    def _normal_logproba(x, mean, logstd, std=None):
        if std is None:
            std = torch.exp(logstd)

        std_sq = std.pow(2)
        logproba = - 0.5 * math.log(2 * math.pi) - logstd - (x - mean).pow(2) / (2 * std_sq)
        return logproba.sum(-1)

    def get_logproba(self, states, actions):
        """
        return probability of chosen the given actions under corresponding states of current network
        :param states: Tensor
        :param actions: Tensor
        """
        action_mean, action_logstd = self._forward_actor(states)
        logproba = self._normal_logproba(actions, action_mean, action_logstd)
        return logproba

    def train(self, msg, clip_now, lr_now):
        for g in self.optimizer.param_groups:
            g['lr'] = lr_now
        # batch > 1
        minibatch_states = torch.stack([msg[i][0][0] for i in range(len(msg))])
        minibatch_actions = torch.stack([msg[i][0][1] for i in range(len(msg))])
        minibatch_oldlogproba = torch.stack([msg[i][0][2] for i in range(len(msg))])
        minibatch_advantages = torch.stack([msg[i][0][3] for i in range(len(msg))])
        minibatch_returns = torch.stack([msg[i][0][4] for i in range(len(msg))])
        minibatch_newlogproba = self.get_logproba(minibatch_states, minibatch_actions)
        minibatch_newvalues = self._forward_critic(minibatch_states).squeeze(-1)

        ratio =  torch.exp(minibatch_newlogproba - minibatch_oldlogproba)
        surr1 = ratio * minibatch_advantages
        surr2 = ratio.clamp(1 - clip_now, 1 + clip_now) * minibatch_advantages
        loss_surr = - torch.mean(torch.min(surr1, surr2))

        # not sure the value loss should be clipped as well 
        # clip example: https://github.com/Jiankai-Sun/Proximal-Policy-Optimization-in-Pytorch/blob/master/ppo.py
        # however, it does not make sense to clip score-like value by a dimensionless clipping parameter
        # moreover, original paper does not mention clipped value 
        if self.lossvalue_norm:
            minibatch_return_6std = 6 * minibatch_returns.std()
            loss_value = torch.mean((minibatch_newvalues - minibatch_returns).pow(2)) / minibatch_return_6std
        else:
            loss_value = torch.mean((minibatch_newvalues - minibatch_returns).pow(2))

        loss_entropy = torch.mean(torch.exp(minibatch_newlogproba) * minibatch_newlogproba)

        total_loss = loss_surr + self.loss_coeff_value * loss_value + self.loss_coeff_entropy * loss_entropy
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss, loss_surr, loss_value, loss_entropy