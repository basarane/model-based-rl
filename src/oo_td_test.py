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

from envs.gym_env import get_env
from env_model.model import *
import keras.backend as K
import tensorflow as tf 
from nets.net import init_nn_library
from agents.agent import VAgent

arguments = vars(args)

init_nn_library(True, "1")

#env = gym_env(args.game)
env = get_env(args.game, args.atari, args.env_transforms)

viewer = None
if args.enable_render:
	viewer = EnvViewer(env, args.render_step, 'human')

envOps = EnvOps(env.observation_space.shape, env.action_space.n, 0)
print(env.observation_space.low)
print(env.observation_space.high)

env_model = globals()[args.env_model](envOps)
env_model.model.load_weights(args.env_weightfile)

v_model = globals()[args.vmodel](envOps)

weight_files = []
if len(args.load_weightfile) == 1:
	weight_files = [args.load_weightfile]
else:
	idxs = range(int(args.load_weightfile[1]), int(args.load_weightfile[3]), int(args.load_weightfile[2]))
	weight_files = [args.load_weightfile[0] + str(I) + '.h5' for I in idxs]
	
summary_writer = tf.summary.FileWriter(args.logdir, K.get_session().graph) if not args.logdir is None else None
	
sw = SummaryWriter(summary_writer, ['Average reward'])
#sw = SummaryWriter(summary_writer, ['Reward'])

for I, weight_file in enumerate(weight_files): 
	v_model.model.load_weights(weight_file)
	v_agent = VAgent(env.action_space, env_model, v_model, envOps, None, False)
	runner = Runner(env, v_agent, None, 1, max_step=args.max_step, max_episode=args.max_episode)
	runner.listen(v_agent, None)
	if viewer is not None:
		runner.listen(viewer, None)
	runner.run()
	stats = np.array(v_agent.stats['reward'])
	aver_reward = stats[:,1].sum() / stats[-1,0]
	sw.add([aver_reward], I)
	print('{0} / {1}: Aver Reward = {2} '.format(I+1, len(weight_files), aver_reward))
	