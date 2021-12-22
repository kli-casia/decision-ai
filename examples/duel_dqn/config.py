import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ind', type=int, default=1,
                        help='specify env instance num')
    parser.add_argument('--env_id', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffersize', type=int, default=10000)
    parser.add_argument('--epsison_max', type=float, default=1)
    parser.add_argument('--epsilon_min', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_start', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--USE_CUDA', type=bool, default=False)
    parser.add_argument('--frames', type=int, default=600000)
    parser.add_argument('--print_interval', type=int, default=1000)
    parser.add_argument('--update_tar_interval', type=int, default=1000)
    parser.add_argument('--train_interval', type=int, default=2)
    parser.add_argument('--eps_decay', type=int, default=30000)
    parser.add_argument('--priority', type=bool, default=False)
    parser.add_argument('--use_wrapper', type=bool, default=True)
    parser.add_argument('--PATH', type=str, default='./model.pkl')
    parser.add_argument('--mlp', type=bool, default=False)
    parser.add_argument('--reward_threshold', type=float, default=20.0)

    return parser.parse_args()
