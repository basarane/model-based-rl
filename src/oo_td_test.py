import argparse

parser = argparse.ArgumentParser(description='DQN Training')
parser.add_argument('game', type=str, default='Breakout', help='Gym game name')
parser.add_argument('--env-weightfile', type=str, default=None, help='load environment weights')
parser.add_argument('--env-model', type=str, default=None, help='class name of environment model')
parser.add_argument('--logdir', type=str, default=None, help='Logdir for tensorboard')
parser.add_argument('--max-step', type=int, default=int(1e10), help='max step')
parser.add_argument('--max-episode', type=int, default=int(10), help='max episode')
parser.add_argument('--test-epsilon', type=float, default=0.05, help='epsilon for testing')
parser.add_argument('--load-weightfile', type=str, nargs='*', default=None, help='load initial weights')
parser.add_argument('--atari', type=bool, default=False, help='true if env is atari game')
parser.add_argument('--env-transforms', type=str, nargs='*', default=[], help='apply the environment transforms')
parser.add_argument('--enable-render', type=bool, default=False, help='Enable render')
parser.add_argument('--render-step', type=int, default=4, help='render step')
parser.add_argument('--vmodel', type=str, default='V Model', help='class name for v-model')

args = parser.parse_args()

from algo.td import run_td_test

arguments = vars(args)

stats = run_td_test(**arguments)
print(stats)