from msagent import Worker, service, logger, event_handler
from msagent.utils import service_utils
import argparse

from duel_dqn import DQN_Agent, epsilon_by_frame


class Actor(Worker):
    def __init__(self, **kwargs):
        super().__init__()
        self.policy = None
        self.eval_cnt = 1

    @service()
    def policy_eval(self, msg):
        if self.policy is not None:
            epsilon = epsilon_by_frame(self.eval_cnt)
            state_tensor = self.policy.observe(msg)
            action = self.policy.act(state_tensor, epsilon)
            self.eval_cnt += 1
            return action

    @event_handler('policy_update')
    def handler(self, model):
        if isinstance(model, argparse.Namespace):
            args = model
            self.policy = DQN_Agent(input_shape=args.input_shape,
                                    action_space=args.action_space,
                                    USE_CUDA=args.USE_CUDA,
                                    lr=args.learning_rate,
                                    priority=args.priority,
                                    mlp=args.mlp)
        else:
            self.policy.DQN.load_state_dict(model)
        logger.info('recv msg, update policy')
