from msagent import remote, logger
import config
import argparse
import gym
import numpy as np
import time


@remote
def policy_init():
    pass


@remote
def policy_eval():
    pass


@remote
def policy_train():
    pass


def env_run(args, conf):
    env = gym.make(conf.task)
    conf.state_shape = env.observation_space.shape or env.observation_space.n
    conf.action_shape = env.action_space.shape or env.action_space.n
    conf.action_space = env.action_space
    conf.max_action = env.action_space.high[0]
    conf.exploration_noise = conf.exploration_noise * conf.max_action
    print("Observations shape:", conf.state_shape)
    print("Actions shape:", conf.action_shape)
    print("Action range:", np.min(env.action_space.low),
          np.max(env.action_space.high))
    # seed
    np.random.seed(conf.seed)
    # model
    policy_init(conf)

    _transition_keys = ['obs', 'act', 'rew', 'obs_next', 'done', 'info']
    transition = dict.fromkeys(_transition_keys)
    transition.update(obs=env.reset())

    total_step = round(conf.epoch * conf.step_per_epoch / conf.env_num)
    step_count, step_per_train = 0, conf.step_per_train
    trj = []
    while True:
        obs = transition['obs']
        acts = policy_eval(obs, exploration=True)
        if acts:
            act, act_remap = acts
            logger.info(f'output act {act}')
            obs_next, rew, done, info = env.step(act_remap)
            transition.update(
                act=act,
                obs_next=obs_next,
                rew=rew,
                done=done,
                info=info)
            trj.append(transition.copy())
            if done:
                transition.update(obs=env.reset())
            else:
                transition.update(obs=obs_next)
            step_count += 1
            if step_count % step_per_train == 0:
                policy_train(trj)
                trj = []
            if step_count >= total_step:
                break
        else:
            time.sleep(1)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ind',
        type=int,
        default=1,
        help='specify env instance num')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    conf = config.POLICY_ARGS
    env_run(args, conf)
