from msagent import Worker, service, logger, EventDispatcher
import argparse
import torch
from dqn_pri import DQN_Agent, epsilon_by_frame
import time


class Learner(Worker):
    def __init__(self, **kwargs):
        super().__init__()
        self.train_cnt = 0
        self.policy = None
        self.ed = EventDispatcher()
        self.initime = self.endtime = time.time()
        self.reward_largest = 10

    @service
    def make_model(self, args):
        self.policy = DQN_Agent(
            input_shape=args.input_shape,
            action_space=args.action_space,
            USE_CUDA=args.USE_CUDA,
            lr=args.learning_rate,
            priority=args.priority,
            mlp=args.mlp,
        )
        self.args = args
        self.ed.fire("policy_update", payload=self.args)

    @service(response=False)
    def policy_train(self, msg):
        frame, action, reward, next_frame, done = msg
        state = self.policy.observe(frame)
        next_state = self.policy.observe(next_frame)
        self.policy.memory_buffer.push(state, action, reward, next_state, done)

        if (
            self.policy.memory_buffer.size() >= self.args.learning_start
            and self.train_cnt % self.args.train_interval == 0
        ):
            logger.info('learn once')
            loss = self.policy.learn_from_experience(self.args.batch_size)

            if self.train_cnt % 20 == 0:
                policy = self.policy.DQN.state_dict()
                self.ed.fire("policy_update", payload=policy)

        if self.train_cnt % self.args.update_tar_interval == 0:
            policy = self.policy.DQN.state_dict()
            self.policy.DQN_target.load_state_dict(policy)
            logger.info('update policy model')
            # self.ed.fire("policy_update", payload=policy)

        
            if self.reward_largest < reward:
                torch.save(self.policy.DQN.state_dict(), self.args.PATH)
                self.reward_largest = reward

        self.train_cnt += 1
        self.endtime = time.time()
