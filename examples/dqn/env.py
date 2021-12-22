from msagent import remote, logger
import numpy as np
import torch
from msagent.utils.atari_wrappers import make_atari, wrap_Framestack
import config
import gym
import os
import time
from os.path import join as joindir
from os import makedirs as mkdir
import pandas as pd

logger.set_level("info")

RESULT_DIR = joindir("../result", ".".join(__file__.split(".")[:-1]))
mkdir(RESULT_DIR, exist_ok=True)


@remote
def policy_eval():
    pass


@remote
def make_model():
    pass


@remote
def policy_train():
    pass


def ensured_policy_eval(state):
    while True:
        data = policy_eval(state)
        if data is not None:
            return data
        else:
            time.sleep(1)


args = config.get_args()
DIR = os.path.abspath(os.path.dirname(__file__) +
                      os.path.sep + "dqn_model.pkl")

env = make_atari(args.env_id) if args.use_wrapper else gym.make(args.env_id)
env = wrap_Framestack(
    env, scale=False, frame_stack=4) if args.use_wrapper else env
frame = env.reset()

args.USE_CUDA = torch.cuda.is_available()
args.action_space = env.action_space
args.input_shape = (
    frame._force().transpose(2, 0, 1).shape
    if not args.mlp
    else env.observation_space.shape
)
args.PATH = DIR

make_model(args)

print_interval = 1000
episode_num = 0
episode_reward = 0
all_rewards = []
reward_record = []
capacity = []
losses = []

frames = 20000000
# avg_reward = 0
start = g_start = time.time()
for i in range(frames):
    msg = frame
    action = ensured_policy_eval(msg)
    next_frame, reward, done, info = env.step(action)
    episode_reward += reward
    msg = [frame, action, reward, next_frame, done]
    loss = policy_train(msg)
    losses.append(loss)
    frame = next_frame
    if i % print_interval == 0:
        logger.info(
            "frames: {}, reward: {:.2f}, episode: {}".format(
                i, np.mean(all_rewards[-10:]), episode_num
            )
        )
        end = time.time()
        print("所需要的时间:{}".format(end - start))
        C = print_interval / (end - start)
        print("吞吐量是:{} bps".format(C))
        capacity.append({"steps": i, "capacity": C})
        capacity_pd = pd.DataFrame(capacity)
        capacity_pd.to_csv(
            joindir(RESULT_DIR, "capacity-record-breakoout.csv"))
        record_pd = pd.DataFrame(reward_record)
        record_pd.to_csv(
            joindir(RESULT_DIR, "dqn-record-{}.csv".format(args.env_id)))
        start = time.time()
    # if avg_reward > args.reward_threshold:
    #         print ("终止训练！")
    #         break
    if done:
        frame = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0
        episode_num += 1
        avg_reward = float(np.mean(all_rewards[-100:]))
        reward_record.append(
            {"episode": episode_num, "meanreward": avg_reward})
g_end = time.time()
print("训练一百万次所需要的时间是: {} hours".format((g_end - g_start)/3600))