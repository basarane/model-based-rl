from envs.gym_env import get_env
from env_model.model import *
import tensorflow.keras.backend as K
import tensorflow as tf 
from nets.net import init_nn_library
from agents.agent import VAgent
from utils.misc import ParameterDecay
from utils.viewer import EnvViewer, save_image

def run_td(**kargs):
    debug = False
    if kargs['output_dir'] is None and kargs['logdir'] is not None:
        kargs['output_dir'] = kargs['logdir']

    from collections import namedtuple
    args = namedtuple("TDParams", kargs.keys())(*kargs.values())

    target_network_update = ParameterDecay(args.target_network_update)
    
    if 'dont_init_tf' in kargs and not kargs['dont_init_tf']:
        init_nn_library(True, "1")

    env = get_env(args.game, args.atari, args.env_transforms, kargs['monitor_dir'] if 'monitor_dir' in kargs else None)

    envOps = EnvOps(env.observation_space.shape, env.action_space.n, args.learning_rate)
    #print(env.observation_space.low)
    #print(env.observation_space.high)

    env_model = globals()[args.env_model](envOps)
    if args.env_weightfile is not None:
        env_model.model.load_weights(args.env_weightfile)

    v_model = globals()[args.vmodel](envOps)

    import numpy as np
    td_model = TDNetwork(env_model.model, v_model, envOps, False, kargs['derivative_coef'] if 'derivative_coef' in kargs else 0)

    summary_writer = tf.summary.FileWriter(args.logdir, K.get_session().graph) if not args.logdir is None else None
    sw = SummaryWriter(summary_writer, ['Loss'])

    if args.load_trajectory is not None:
        from utils.trajectory_utils import TrajectoryLoader
        traj = TrajectoryLoader(args.load_trajectory)

    from utils.network_utils import NetworkSaver
    network_saver = NetworkSaver(args.save_freq, args.logdir, v_model.model)

    import scipy.stats as stats

    td_exponent = ParameterDecay(kargs['td_exponent'] if 'td_exponent' in kargs and kargs['td_exponent'] is not None else 2)

    #from tensorflow.keras.utils import plot_model
    #plot_model(td_model.td_model, to_file='td_model.png')
    td_model.td_model.summary()
    
    print('TDNetwork Layers')
    for layer_idx, layer in enumerate(td_model.td_model.layers):
        print('Layer ', layer_idx, layer.name, layer.shape if hasattr(layer, "shape") else layer.input_shape)
    
    for I in range(args.max_step):
        #batch = np.random.uniform([-4.8, -5, -0.48, -5], [4.8, 5, 0.48, 5], size=(args.batch_size,4))
        #lower, upper = -1, 1
        #mu, sigma = 0.5, 0.4
        #X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        #samples = np.random.uniform([-1], [1], size=(5000,1))
        if hasattr(env_model, "get_samples"):
            samples = env_model.get_samples(args.sample_count)
        else:
            samples = np.random.uniform(args.smin, args.smax, size=(args.sample_count,len(args.smin)))
        res = td_model.test(samples)
        if isinstance(res, (list,)):
            res = res[0]
        td_errors = res.flatten()
        props = np.abs(td_errors)
        #props = np.multiply(props, props)
        props = np.power(props, td_exponent())
        props = props / props.sum()
        count = 0
        if isinstance(samples, list):
            count = samples[0].shape[0]
        else:
            count = samples.shape[0]
        idxs = np.random.choice(count, args.batch_size, False, props)
        batch = {
            #'current': np.random.uniform([-1], [1], size=(args.batch_size,1))
            #'current': X.rvs((args.batch_size,1))
            'current': [a[idxs] for a in samples] if isinstance(samples, list) else samples[idxs]
        }
        #batch = traj.sample(args.batch_size)
        
        if debug:
            # @ersin - freeway icin test kodu
            old_loss = td_model.test(batch['current'])
            print(I, old_loss.flatten())
            decoder_input = [batch['current'][1].astype(np.float32), batch['current'][0].astype(np.float32)]
            decoded_output = env_model.model_decoder.predict_on_batch(decoder_input)
            save_image(decoded_output, args.batch_size, f'{kargs["output_dir"]}/{I}_sample')
        
        loss = td_model.train(batch['current'])
        
        #@ersin - tests
        if debug and False:
            save_image(batch['current'][0], args.batch_size, f'{kargs["output_dir"]}/{I}_current_cars')
            save_image(batch['current'][1], args.batch_size, f'{kargs["output_dir"]}/{I}_current_tavuks')
            save_image(batch['current'][2], args.batch_size, f'{kargs["output_dir"]}/{I}_current_cross')
            save_image(batch['current'][3], args.batch_size, f'{kargs["output_dir"]}/{I}_current_carpisma')

            next_state = env_model.predict_next(batch['current'])

            for J in range(3):
                save_image(next_state[0+J*4], args.batch_size, f'{kargs["output_dir"]}/{I}_next_{J}_cars')
                save_image(next_state[1+J*4], args.batch_size, f'{kargs["output_dir"]}/{I}_next_{J}_tavuks')
                save_image(next_state[2+J*4], args.batch_size, f'{kargs["output_dir"]}/{I}_next_{J}_cross')
                save_image(next_state[3+J*4], args.batch_size, f'{kargs["output_dir"]}/{I}_next_{J}_carpisma')

        print(loss)
        if td_model.include_derivative:
            loss = loss[0]
        sw.add([loss], I)
        network_saver.on_step()
        td_exponent.on_step()
        target_network_update.on_step()
        if target_network_update.is_step() == 0:
            td_model.v_model_eval.set_weights(td_model.v_model.get_weights())
        

def run_td_test(**kargs):
    if ('output_dir' not in kargs or kargs['output_dir'] is None) and kargs['logdir'] is not None:
        kargs['output_dir'] = kargs['logdir']

    from collections import namedtuple
    args = namedtuple("TDTestParams", kargs.keys())(*kargs.values())

    if 'dont_init_tf' in kargs and not kargs['dont_init_tf']:
        init_nn_library(True, "1")

    #env = gym_env(args.game)
    print('Monitor dir', kargs['monitor_dir'] if 'monitor_dir' in kargs else None)
    env = get_env(args.game, args.atari, args.env_transforms, kargs['monitor_dir'] if 'monitor_dir' in kargs else None)

    viewer = None
    if args.enable_render:
        viewer = EnvViewer(env, args.render_step, 'human')

    envOps = EnvOps(env.observation_space.shape, env.action_space.n, 0)
    #print(env.observation_space.low)
    #print(env.observation_space.high)

    env_model = globals()[args.env_model](envOps)
    if args.env_weightfile is not None:
        env_model.model.load_weights(args.env_weightfile)

    v_model = globals()[args.vmodel](envOps)

    weight_files = []
    if not isinstance(args.load_weightfile,list):
        weight_files = [(args.load_weightfile,0)]
    else:
        idxs = range(int(args.load_weightfile[1]), int(args.load_weightfile[3]), int(args.load_weightfile[2]))
        weight_files = [(args.load_weightfile[0] + str(I) + '.h5',I) for I in idxs]
        
    summary_writer = tf.summary.FileWriter(args.logdir, K.get_session().graph) if not args.logdir is None else None
        
    sw = SummaryWriter(summary_writer, ['Average reward', 'Total reward'])
    #sw = SummaryWriter(summary_writer, ['Reward'])

    stats = {
        'reward': []
    }
    for I, weight_file_info in enumerate(weight_files): 
        weight_file = weight_file_info[0]
        total_step_count = weight_file_info[1]
        v_model.model.load_weights(weight_file)
        v_agent = VAgent(env.action_space, env_model, v_model, envOps, None, False)
        runner = Runner(env, v_agent, None, 1, max_step=args.max_step, max_episode=args.max_episode)
        runner.listen(v_agent, None)
        if viewer is not None:
            runner.listen(viewer, None)
        runner.run()
        tmp_stats = np.array(v_agent.stats['reward'])
        total_reward = tmp_stats[:,1].sum()
        total_reward = total_reward / args.max_episode
        aver_reward = total_reward / tmp_stats[-1,0]
        sw.add([aver_reward, total_reward], I)
        stats['reward'].append((total_step_count, total_reward))
        print('{0} / {1}: Aver Reward per step = {2}, Aver Reward per espisode = {3}'.format(I+1, len(weight_files), aver_reward, total_reward))		
    return stats