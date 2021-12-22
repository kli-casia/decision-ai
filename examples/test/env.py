from msagent import remote, logger, config
import time
import numpy as np


@remote
def policy_eval():
    pass


@remote
def make_model():
    pass


@remote
def policy_train():
    pass


step_cnt = 0
state = np.ones([50, 50], dtype=np.float32)

make_model('env info')

while True:
    action = policy_eval(state)
    print("fuck")
    print(type(action))
    print(action)
    policy_train(state)
    step_cnt += 1
    print("env step", step_cnt)
