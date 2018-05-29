import argparse

parser = argparse.ArgumentParser(description='DQN Training')
parser.add_argument('game', type=str, default='Breakout', help='Gym game name')
parser.add_argument('--mode', type=str, default="train", help='mode: train or test')
parser.add_argument('--env-model', type=str, default=None, help='class name of environment model')
parser.add_argument('--env-weightfile', type=str, default=None, help='load environment weights')
parser.add_argument('--output-dir', type=str, default=None, help='output directory')
parser.add_argument('--logdir', type=str, default=None, help='Logdir for tensorboard')
parser.add_argument('--learning-rate', type=float, default=0.00025, help='learning rate')
parser.add_argument('--batch-size', type=int, default=int(32), help='batch size')
parser.add_argument('--max-step', type=int, default=int(1e10), help='max step')
parser.add_argument('--max-episode', type=int, default=int(1e10), help='max episode')
parser.add_argument('--save-interval', type=int, default=10000, help='save interval')
parser.add_argument('--target-network-update', type=int, default=1000, help='target network update feq')
parser.add_argument('--update-frequency', type=int, default=4, help='training update frequency')
parser.add_argument('--replay-buffer-size', type=int, default=int(1e6), help='the number of transitions in replay buffer')
parser.add_argument('--replay-start-size', type=int, default=int(50000), help='replay start size')
parser.add_argument('--test-epsilon', type=float, default=0.05, help='epsilon for testing')
parser.add_argument('--load-weightfile', type=str, default=None, help='load initial weights')
parser.add_argument('--atari', type=bool, default=False, help='true if env is atari game')
parser.add_argument('--env-transforms', type=str, nargs='*', default=[], help='apply the environment transforms')
parser.add_argument('--dueling-dqn', type=bool, default=False, help='Dueling dqn')
parser.add_argument('--enable-render', type=bool, default=False, help='Enable render')
parser.add_argument('--model', type=str, default='DQNModel', help='class name for q-model')
parser.add_argument('--double-dqn', type=bool, default=False, help='Use double dqn')
parser.add_argument('--load-trajectory', type=str, default=None, help='load sample trajectories from this file')
parser.add_argument('--vmodel', type=str, default='V Model', help='class name for v-model')
parser.add_argument('--save-freq', type=int, default=5000, help='save network after this many batches')
parser.add_argument('--egreedy-props', type=float, nargs='*', default=[1], help='multiple egreedy props')
parser.add_argument('--egreedy-final', type=float, nargs='*', default=[0.1], help='multiple egreedy final exploration')
parser.add_argument('--egreedy-final-step', type=int, nargs='*', default=[int(1e6)], help='multiple egreedy final step')
parser.add_argument('--egreedy-decay', type=float, default=1, help='exponential decay rate for egreedy')

args = parser.parse_args()

from algo.td_realtime import run_td_realtime

arguments = vars(args)

runner, _ = run_td_realtime(**arguments)

runner.run()