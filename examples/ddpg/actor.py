from msagent import Worker, service, logger, event_handler, EventDispatcher
import argparse
import torch
import numpy as np
from model import to_numpy


class Actor(Worker):
    ed = EventDispatcher()

    def __init__(self,):
        args = get_args()
        super().__init__(service_name='actor')
        self.policy = None

    @service
    def policy_eval(self, obs, exploration=False):
        if self.policy is not None:
            with torch.no_grad():
                if len(obs.shape) == 1:
                    obs = np.expand_dims(obs, 0)
                act = self.policy.actor(obs).flatten()
            act = to_numpy(act)
            if exploration:
                act = self.policy.exploration_noise(act)
            act_remap = self.policy.map_action(act)
            return act, act_remap
        else:
            print("actor model is not initialized!")

    @event_handler('policy_update')
    def handler(self, model):
        logger.info("recv broadcast, update policy")
        if not self.policy:
            self.policy = model
        else:
            self.policy.load_state_dict(model)


def get_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args

