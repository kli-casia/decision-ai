from msagent import Worker, service, logger, EventDispatcher
import torch
import numpy as np
import gym
import datetime
import os
import argparse
from ddpg import Buffer
from ddpg import build_model, preprocess_buffer, test_episode
import config


class Learner(Worker):
    ed = EventDispatcher()

    def __init__(self, **kwargs):
        args = get_args(config.POLICY_ARGS)
        super().__init__(service_name='learner')
        self.policy = None

        # here env can not find service below, wait...
        self.env = gym.make(args.task)
        self.buffer = Buffer(maxsize=args.buffer_size)
        start_steps = args.start_timesteps
        preprocess_buffer(self.buffer, self.env, start_steps)

    @service
    def policy_init(self, args):
        if not self.policy:
            self.conf = args
            self.train_count = 0
            self.test_count = 0
            self.best_reward, self.best_epoch = None, 0
            # seed
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            # checkpoint
            t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
            log_file = f'seed_{args.seed}_{t0}-{args.task.replace("-", "_")}_ddpg'
            self.log_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                         args.logdir, args.task, 'ddpg', log_file)

            # policy
            self.policy = build_model(args.state_shape, args.action_shape, args.hidden_sizes,
                                      args.actor_lr, args.critic_lr, args.action_space, args.max_action,
                                      args.device, args.tau, args.gamma, args.exploration_noise)
            logger.info("Policy Initialized")
            self.ed.fire('policy_update', payload=self.policy)
            logger.info("send broadcast, update policy")

            if not self.best_reward:
                test_result = test_episode(
                    self.env, self.policy, args.episode_per_test)
                self.best_reward = test_result['rew']
                logger.info(f'first test with reward {self.best_reward:.3f}')
        else:
            logger.info("policy has been initialized by some env")

    @service(response=False)
    def policy_train(self, msg):
        trj = msg
        len_trj = len(trj)
        for transition in msg:
            self.buffer.append(transition)
        batch_size = self.conf.batch_size
        buffer = self.buffer
        for _ in range(len_trj):
            loss = self.policy.update(batch_size, buffer)
            self.train_count += 1
            print(f'n_train: {self.train_count}, loss: {loss}')
            if self.train_count % self.conf.step_per_epoch == 0:
                self.ed.fire('policy_update', payload=self.policy.state_dict())
                logger.info("send broadcast, update policy")

                n_episode = self.conf.episode_per_test
                test_result = test_episode(self.env, self.policy, n_episode)
                self.test_count += 1
                rew, rew_std = test_result['rew'], test_result['rew_std']
                len_, len_std = test_result['len'], test_result['len_std']
                if rew > self.best_reward:
                    self.best_reward, self.best_epoch = rew, self.test_count
                    best_policy = {"epoch": self.best_epoch, "rew": rew, "rew_std": rew_std,
                                   "len": len_, "len_std": len_std, "policy": self.policy.state_dict()}
                    torch.save(
                        best_policy, os.path.join(
                            self.log_path, 'policy.pth'))
                    logger.info("best policy updated")
                logger.info(f"TEST: epoch: {self.test_count}, reward: {rew:.3f} ({rew_std:.3f})"
                            f"len: {int(len_)} ({len_std:.3f})"
                            f"best_rew(epoch): {self.best_reward:.3f}({self.best_epoch})")


def get_args(conf):
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default=conf.task)
    parser.add_argument('--buffer-size', type=int, default=conf.buffer_size)
    parser.add_argument(
        '--start_timesteps',
        type=int,
        default=conf.start_timesteps)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    conf = config.POLICY_ARGS
    args = get_args(conf)
    learner = Learner(args)
    learner.run()
