env_name = 'Walker2d-v2' # can set any mujoco environment, eg.'Walker2d-v2', 'Swimmer-v2', 'Hopper-v2', 'Humanoid-v2', 'HalfCheetah-v2', 'Reacher-v2'
env_num = 0
seed = 1234
num_episode = 2000
batch_size = 2048
max_step_per_round = 2000
gamma = 0.995
lamda = 0.97
log_num_episode = 1
num_epoch = 10
minibatch_size = 256
clip = 0.2
loss_coeff_value = 0.5
loss_coeff_entropy = 0.01
lr = 3e-4
# tricks
schedule_adam = 'linear'
schedule_clip = 'linear'
layer_norm = True
state_norm = True
advantage_norm = True
lossvalue_norm = True
EPS = 1e-10