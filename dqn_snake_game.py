import argparse
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from snake_game import SnakeGame


def train_dqn():
    from dqn import DqnTrain

    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--save_rate', type=int, default=25)
    parser.add_argument('--number_hide_layers', type=int, default=2)
    parser.add_argument('--number_neural_by_layer', type=int, default=64)
    parser.add_argument('--buffer_size', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epsilon_decay', type=float, default=3 * 1e-4)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--replace', type=int, default=100)

    args, _ = parser.parse_known_args()
    dqn = DqnTrain(**vars(args))
    dqn.train()


def run_dqn():
    from dqn import DqnRun

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--file_name', type=str, default='output')

    args, _ = parser.parse_known_args()
    dqn = DqnRun(args.model_name)
    dqn.run(args.save, args.file_name)


def play():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size_field', type=int, default=18)
    parser.add_argument('--time_speed', type=float, default=0.1)

    args, _ = parser.parse_known_args()
    snake_game = SnakeGame(**vars(args))
    snake_game.play_game()


def get_action_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str)
    action_arg, _ = parser.parse_known_args()

    return action_arg


if __name__ == '__main__':
    if len(sys.argv) > 1:
        arg = get_action_argument()
        globals()[arg.action]()
    else:
        exit()