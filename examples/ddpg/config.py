import torch


class objdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


POLICY_DICT = {'task': 'Ant-v3',
               'seed': 0,
               'buffer_size': 1000000,
               'hidden_sizes': [256, 256],
               'actor_lr': 1e-3,
               'critic_lr': 1e-3,
               'gamma': 0.99,
               'tau': 0.005,
               'batch_size': 256,
               'start_timesteps': 25000,
               'epoch': 500,
               'step_per_epoch': 5000,
               'step_per_train': 10,
               'episode_per_test': 2,
               'exploration_noise': 0.1,
               'env_num': 2,
               'logdir': 'checkpoint',
               'device': 'cuda' if torch.cuda.is_available() else 'cpu'
               }

POLICY_ARGS = objdict(POLICY_DICT)
