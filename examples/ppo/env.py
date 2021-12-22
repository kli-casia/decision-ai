from msagent import remote, logger
from os.path import join as joindir
from os import makedirs as mkdir
from torch import Tensor
import pandas as pd
import numpy as np
import time
import gym

from ppo import PPO, ZFilter
import config as conf


@remote
def make_model(num_inputs, num_outputs):
    pass


@remote
def policy_eval(msg):
    pass


@remote
def policy_train(msg, clip=conf.clip, lr=conf.lr):
    pass


RESULT_DIR = joindir("../result", ".".join(__file__.split(".")[:-1]))
mkdir(RESULT_DIR, exist_ok=True)


def ensured_policy_eval(state):
    while True:
        data = policy_eval(state)
        if data is not None:
            return data
        else:
            time.sleep(1)


def train(conf):
    env = gym.make(conf.env_name)
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    env.seed(conf.seed)

    make_model(num_inputs, num_actions)

    algo = PPO(conf)
    running_state = ZFilter((num_inputs,), clip=5.0)

    # record average 1-round cumulative reward in every episode
    reward_record = []
    global_steps = 0

    lr_now = conf.lr
    clip_now = conf.clip

    for i_episode in range(conf.num_episode):
        # step1: perform current policy to collect trajectories
        # this is an on-policy method!
        num_steps = 0
        reward_list = []
        len_list = []
        while num_steps < conf.batch_size:
            state = env.reset()
            if conf.state_norm:
                state = running_state(state)
            reward_sum = 0
            for t in range(conf.max_step_per_round):
                state = Tensor(state).unsqueeze(0)
                action_mean, action_logstd, value = ensured_policy_eval(state)
                action, logproba = algo.select_action(action_mean, action_logstd)
                action = action.data.numpy()[0]
                logproba = logproba.data.numpy()[0]
                next_state, reward, done, _ = env.step(action)
                reward_sum += reward
                if conf.state_norm:
                    next_state = running_state(next_state)
                mask = 0 if done else 1

                algo.memory.push(
                    state, value, action, logproba, mask, next_state, reward
                )

                if done:
                    break

                state = next_state

            num_steps += t + 1
            global_steps += t + 1
            reward_list.append(reward_sum)
            len_list.append(t + 1)
        
        reward_record.append(
            {
                "episode": i_episode,
                "steps": global_steps,
                "meanepreward": np.mean(reward_list),
                "meaneplen": np.mean(len_list),
            }
        )

        algo.process_traj()

        for i_epoch in range(
            int(conf.num_epoch * algo.batch_size / conf.minibatch_size)
        ):
            # sample from current batch
            # 
            msg = algo.get_sample()
            total_loss, loss_surr, loss_value, loss_entropy = policy_train(
                msg, clip=clip_now, lr=lr_now
            )

        if conf.schedule_clip == "linear":
            ep_ratio = 1 - (i_episode / conf.num_episode)
            clip_now = conf.clip * ep_ratio

        if conf.schedule_adam == "linear":
            ep_ratio = 1 - (i_episode / conf.num_episode)
            lr_now = conf.lr * ep_ratio

        if i_episode % conf.log_num_episode == 0:
            logger.info(
                "Finished episode: {} Reward: {:.4f} total_loss = {:.4f} = {:.4f} + {} * {:.4f} + {} * {:.4f}".format(
                    i_episode,
                    reward_record[-1]["meanepreward"],
                    total_loss.data,
                    loss_surr.data,
                    conf.loss_coeff_value,
                    loss_value.data,
                    conf.loss_coeff_entropy,
                    loss_entropy.data,
                )
            )
            logger.info("-----------------")

    return reward_record


if __name__ == "__main__":
    reward_record = train(conf)
    reward_record = pd.DataFrame(reward_record)
    reward_record.to_csv(
        joindir(RESULT_DIR, "ppo-record-{}-{}.csv".format(conf.env_name, conf.env_num))
    )
