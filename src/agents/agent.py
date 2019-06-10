from runner.runner import RunnerListener
import random
import numpy as np
import copy

from PIL import Image
import sys

from utils.summary_writer import SummaryWriter
from tensorflow.keras.utils import to_categorical
from utils.viewer import save_image

class Agent(object):
    def __init__(self, action_space, ops):
        self.action_space = action_space
        self.ops = ops
        self.stats = {}
    def act(self, observation):
        raise NotImplementedException()
    def add_stat(self, name, val):
        if not name in self.stats:
            self.stats[name] = []
        self.stats[name].append(val)

class RandomAgent(Agent):
    def __init__(self, action_space, ops):
        super(RandomAgent, self).__init__(action_space, ops)
    def act(self, observation):
        return self.action_space.sample()

class EGreedyOps(object):
    def __init__(self):
        self.test_epsilon = 0.05
        self.mode = "train"
        self.REPLAY_START_SIZE = 50000
        self.INITIAL_EXPLORATION = 1.0
        self.FINAL_EXPLORATION = 0.1
        self.FINAL_EXPLORATION_FRAME = 1e6
        
class EGreedyAgent(Agent,RunnerListener):
    def __init__(self, action_space, ops, agent):
        super(EGreedyAgent, self).__init__(action_space, ops)
        self.total_step_count = 0
        self.random_agent = RandomAgent(action_space, ops)
        self.agent = agent
    def act(self, observation):
        action = self.random_agent.act(observation)
        if (self.total_step_count > self.ops.REPLAY_START_SIZE) or self.ops.mode == "test":
            epsilon = (self.ops.INITIAL_EXPLORATION-self.ops.FINAL_EXPLORATION) * max(self.ops.FINAL_EXPLORATION_FRAME-self.total_step_count, 0) / (self.ops.FINAL_EXPLORATION_FRAME-self.ops.REPLAY_START_SIZE) + self.ops.FINAL_EXPLORATION
            if self.ops.mode == "test":
                epsilon = self.ops.test_epsilon
            if epsilon < random.random():
                action = self.agent.act(observation)
        return action
    def on_step(self, ob, action, next_ob, reward, done):
        self.total_step_count += 1
        
class EGreedyAgentExp(Agent,RunnerListener):
    def __init__(self, action_space, ops, agent):
        super(EGreedyAgentExp, self).__init__(action_space, ops)
        self.total_step_count = 0
        self.random_agent = RandomAgent(action_space, ops)
        self.agent = agent
        self.epsilon = self.ops.INITIAL_EXPLORATION
    def act(self, observation):
        action = self.random_agent.act(observation)
        if (self.total_step_count > self.ops.REPLAY_START_SIZE) or self.ops.mode == "test":
            epsilon = self.epsilon
            if self.ops.mode == "test":
                epsilon = self.ops.test_epsilon
            if epsilon < random.random():
                action = self.agent.act(observation)
        return action
    def on_step(self, ob, action, next_ob, reward, done):
        self.total_step_count += 1
        if self.total_step_count > self.ops.REPLAY_START_SIZE:
            self.epsilon *= self.ops.DECAY
        
class MultiEGreedyAgent(Agent,RunnerListener):
    def __init__(self, action_space, ops, agent, dist_prop=[1], final_exp=[0.1], decay=[1],final_exp_frame=[1e6]):
        super(MultiEGreedyAgent, self).__init__(action_space, ops)
        self.dist_prop = dist_prop
        idx = np.random.choice(range(0, len(dist_prop)), 1, self.dist_prop)[0]
        tmp_ops = copy.copy(ops)
        if decay[idx]<1:
            tmp_ops.DECAY = decay[idx]
            tmp_agent = EGreedyAgentExp(action_space, tmp_ops, agent)
        else:
            tmp_ops.FINAL_EXPLORATION = final_exp[idx]
            tmp_ops.FINAL_EXPLORATION_FRAME = final_exp_frame[idx]
            tmp_agent = EGreedyAgent(action_space, tmp_ops, agent)
        self.tmp_agent = tmp_agent
        #self.greedy_agents = []
        #for x in zip(final_exp, decay):
        #	tmp_ops = copy.copy(ops)
        #	tmp_ops.FINAL_EXPLORATION = x[0]
        #	tmp_ops.DECAY = x[1]
        #	if x[1]<1:
        #		tmp_agent = EGreedyAgentExp(action_space, tmp_ops, agent)
        #	else:
        #		tmp_agent = EGreedyAgent(action_space, tmp_ops, agent)
        #	self.greedy_agents.append(tmp_agent)
    def act(self, observation):
        #agent = np.random.choice(self.greedy_agents, 1, self.dist_prop)
        #return agent[0].act(observation)
        return self.tmp_agent.act(observation)
    def on_step(self, ob, action, next_ob, reward, done):
        self.tmp_agent.on_step(ob, action, next_ob, reward, done)
        
class DqnAgentOps(object):
    def __init__(self):
        self.test_epsilon = 0.05
        self.mode = "train"
        self.double_dqn = False
        self.MINIBATCH_SIZE = 32
        self.DISCOUNT_FACTOR = 0.99
        self.TARGET_NETWORK_UPDATE_FREQUENCY=1e4
        
class DqnAgent(Agent, RunnerListener):
    def __init__(self, action_space, q_model, sampler, rewproc, ops, sw = None, model_eval=None):
        super(DqnAgent, self).__init__(action_space, ops)
        self.q_model = q_model
        if model_eval is not None or self.ops.mode == 'test':
            self.q_model_eval = model_eval
        else:
            self.q_model_eval = q_model.clone()
        self.sampler = sampler
        self.total_step_count = 0
        self.losses = []
        self.rewproc = rewproc
        self.sw = SummaryWriter(sw, ['Episode reward', 'Loss per batch'])
    def act(self, observation):
        prediction = self.q_model.q_value([observation])[0]
        action = np.argmax(prediction)
        return action
        
    def on_step(self, ob, action, next_ob, reward, done):
        self.total_step_count += 1
        if self.sampler is not None and self.ops.mode == "train":
            if self.sampler.has_sample():
                samples = self.sampler.get_sample()
                current_states = [a['current_state'] for a in samples]
                next_states = [a['next_state'] for a in samples]
                
                target = self.q_model.q_value(current_states)
                next_value = self.q_model_eval.q_value(next_states)

                if self.ops.double_dqn:
                    next_best_res = self.q_model.q_value(next_states)
                    best_acts = np.argmax(next_best_res, axis=1)
                else:
                    best_acts = np.argmax(next_value, axis=1)
                
                R = 0
                for I in reversed(range(len(samples))):
                    transition = samples[I]
                    action = transition['action']
                    reward = transition['reward']
                    if self.rewproc is not None:
                        reward = self.rewproc.preprocess(reward)
                    if (self.sampler.nstep() and I==len(samples)-1) or not self.sampler.nstep():
                        R = next_value[I,best_acts[I]]
                    if transition['done']:
                        R = reward
                    else:
                        R = reward + self.ops.DISCOUNT_FACTOR * R   #after double DQN
                    target[I,action] = R
                res = self.q_model.q_update(current_states, target)
                self.losses.append(res)
                self.update_count += 1
            if self.total_step_count % self.ops.TARGET_NETWORK_UPDATE_FREQUENCY == 0 and self.ops.mode == "train":
                self.q_model_eval.set_weights(self.q_model.get_weights())
    def on_episode_start(self):
        self.update_count = 0
    def on_episode_end(self, reward, step_count):
        if len(self.losses)>0:
            x = np.array(self.losses)
            aver_loss = x.sum() / self.update_count
        else:
            aver_loss = 0
        print('Episode end', reward, step_count, self.total_step_count, aver_loss)
        self.update_count = 0
        self.sw.add([reward, aver_loss], self.total_step_count)
        self.losses = []
        self.add_stat('reward', (self.total_step_count, reward))

class ActorCriticAgent(Agent, RunnerListener):
    def __init__(self, action_space, ac_model, sampler, rewproc, ops, sw = None, ac_model_update = None):
        super(ActorCriticAgent, self).__init__(action_space, ops)
        self.ac_model = ac_model
        if ac_model_update is None:
            self.ac_model_update = ac_model
        else:
            self.ac_model_update = ac_model_update
        self.sampler = sampler
        self.total_step_count = 0
        self.losses = []
        self.rewproc = rewproc
        self.sw = SummaryWriter(sw, ['Episode reward', 'Loss Total', 'Loss Actor', 'Loss Critic'])
    def act(self, observation):
        observation = np.array([observation])
        #print(observation.shape, observation)
        prediction = self.ac_model.model_actor.predict(observation)[0]
        action = np.random.choice(range(self.action_space.n), p=prediction)
        return action
        
    def on_step(self, ob, action, next_ob, reward, done):
        self.total_step_count += 1
        if self.sampler is not None and self.ops.mode == "train":
            if self.sampler.has_sample():
                samples = self.sampler.get_sample()
                current_states = [a['current_state'] for a in samples]
                next_states = [a['next_state'] for a in samples]

                #basedir = '/host/PhD/reinforcement-learning/2-cartpole/5-a3c/data/'
                #self.ac_model.model_critic.load_weights(basedir + 'critic0.h5')
                #self.ac_model.model_actor.load_weights(basedir + 'actor0.h5')
                #self.ac_model_update.model_critic.load_weights(basedir + 'critic0.h5')
                #self.ac_model_update.model_actor.load_weights(basedir + 'actor0.h5')
                #dump = np.load(basedir + "dump0.npz")
                #
                #current_states = dump['states']
                #next_states = dump['states']
                #actions = dump['actions']
                #rewards = dump['rewards']
                #print(dump.keys())
                #samples = range(current_states.shape[0])
                
                #target = self.ac_model.q_value(current_states)
                target = self.ac_model.model_critic.predict_on_batch(np.array(current_states))
                next_value = self.ac_model.model_critic.predict_on_batch(np.array(next_states))
                discounted_return = np.zeros((len(samples),), dtype='f')
                
                R = 0
                reward_action = np.zeros((len(samples),self.action_space.n), dtype='f')
                #print(len(samples))
                for I in reversed(range(len(samples))):
                    transition = samples[I]
                    action = transition['action']
                    reward = transition['reward']
                    reward_action[I] = np.array(to_categorical(action,num_classes=self.action_space.n), dtype='f').flatten().squeeze().reshape((1, self.action_space.n))
                    # @ersin - <500 is added for rlcode cartpole test
                    done = transition['done'] #and len(samples)<500
                    
                    #reward_action[I] = actions[I]
                    #reward = rewards[I]
                    #done = I == len(samples)-1
                    
                    if self.rewproc is not None:
                        reward = self.rewproc.preprocess(reward)
                    if (self.sampler.nstep() and I==len(samples)-1) or not self.sampler.nstep():
                        R = next_value[I]
                    
                    if done:
                        R = reward
                    else:
                        R = reward + self.ops.DISCOUNT_FACTOR * R   #after double DQN
                    discounted_return[I] = R
                    
                target = np.reshape(target, (target.shape[0],))
                #print('train_step', current_states, reward_action, discounted_return - target, discounted_return)

                loss1 = self.ac_model_update.model_actor.manual_optimizer([current_states, reward_action, discounted_return - target])
                loss2 = self.ac_model_update.model_critic.manual_optimizer([current_states, discounted_return])
                
                if self.ac_model_update is not self.ac_model:
                    print('different ac3 update model')
                    self.ac_model.set_weights(self.ac_model_update.get_weights())
                #print('R', discounted_return)
                #print(loss1, loss2)
                loss1 = loss1[0] / len(samples)
                loss2 = loss2[0] / len(samples)
                
                #print(dump['values'])
                #print(target)
                #
                #print(dump['discounted_rewards'])
                #print(discounted_return)
                #
                #print(dump['advantages'])
                #print(discounted_return - target)
                #
                #print(dump['loss1'])
                #print(loss1)
                #
                #print(dump['loss2'])
                #print(loss2)
                #
                #sys.exit()
                
                res = [loss1 + loss2, loss1, loss2]
                self.losses.append(res)
                self.update_count += 1
    def on_episode_start(self):
        self.update_count = 0
    def on_episode_end(self, reward, step_count):
        if len(self.losses)>0:
            x = np.array(self.losses)
            aver_loss = x.sum(axis=0) / self.update_count
        else:
            aver_loss = 0
        print('Episode end', reward, step_count, self.total_step_count, aver_loss)
        self.update_count = 0
        self.sw.add([reward, aver_loss[0], aver_loss[1], aver_loss[2]], self.total_step_count)
        self.losses = []
        self.add_stat('reward', (self.total_step_count, reward))
        
from env_model.model import TDNetwork

class VAgent(Agent, RunnerListener):
    def __init__(self, action_space, env_model, v_model, ops, sw = None, verbose = True, sampler = None, target_network_update = 10000):
        super(VAgent, self).__init__(action_space, ops)
        self.env_model = env_model
        self.v_model = v_model
        self.sampler = sampler
        self.sw = SummaryWriter(sw, ['Episode reward', 'Loss per batch'])
        self.total_step_count = 0
        self.verbose = verbose
        self.losses = []
        self.target_network_update = target_network_update
        #if self.ops.mode == "train":
        self.td_model = TDNetwork(self.env_model.model, self.v_model, self.ops)

    def on_step(self, ob, action, next_ob, reward, done):
        self.total_step_count += 1
        if self.sampler is not None and self.ops.mode == "train":
            if self.sampler.has_sample():
                samples = self.sampler.get_sample()
                current_states = [a['current_state'] for a in samples]
                #@ersin - test icin eklendi cikart
                #current_states = np.random.random((64,1))*2-1
                loss = self.td_model.train(np.array(current_states))
                self.losses.append(loss)
                self.update_count += 1
        if self.total_step_count % self.target_network_update == 0 and self.ops.mode == "train":
            self.td_model.v_model_eval.set_weights(self.td_model.v_model.get_weights())
                    

    def act(self, observation):
        next_obs = self.env_model.predict_next(observation)
        #print(next_obs)
        state_count = len(self.env_model.model.inputs)
        prediction = np.zeros(self.action_space.n, dtype='float')
        #print(self.total_step_count)
        for I in range(self.action_space.n):
            #@ersin - normalde rewardi da eklemek gerekir asagidaki gibi, simdilik eskisi gibi birakiyorum
            r = next_obs[self.action_space.n*state_count][0][I]
            done = next_obs[self.action_space.n*state_count+1+I][0][0]
            #print(r, done)
            #@ersin buraya kod eklemisim CartPole icin
            #prediction[I] = (r*100 if r<0 else r) + (1-done)*0.99*self.v_model.v_value(next_obs[I])[0]
            next_v_value = self.v_model.v_value(next_obs[I*state_count:(I+1)*state_count])[0]
            prediction[I] = r + (1-done)*0.99*next_v_value
            #print(I,r,done,next_v_value,(1-done)*0.99*next_v_value,next_obs[I*state_count+2].flatten(),next_obs[I*state_count+3].flatten())
            #prediction[I] = self.v_model.v_value(next_obs[I])[0]
            #decoded_output = self.env_model.model_decoder.predict_on_batch([next_obs[I*state_count+1].astype(np.float32), next_obs[I*state_count].astype(np.float32)])
            #save_image(decoded_output, 1, f'freeway_td/test26/{self.total_step_count}_next_decoded_{I}')
            
        #decoded_output = self.env_model.model_decoder.predict_on_batch([observation[1].astype(np.float32), observation[0].astype(np.float32)])
        #save_image(decoded_output, 1, f'freeway_td/test26/{self.total_step_count}_decoded_observation')
            
        #td_error = self.td_model.test(observation)
        #print(prediction, td_error,observation[2].flatten(),observation[3].flatten())
        
        #v_values = np.zeros((210,))
        #for I in range(210):
        #    cars = np.zeros((1,160,10))
        #    tavuks = np.zeros((1,210,3))
        #    cross_timer = np.zeros((1,7,1))
        #    carpisma_timer = np.zeros((1,13,1))
        #    tavuks[0,I,0] = 1
        #    v_value = self.v_model.v_value([cars, tavuks, cross_timer, carpisma_timer])[0]
        #    v_values[I] = v_value
        #    print(I,v_value)
        #import matplotlib.pyplot as plt
        #plt.plot(np.array(range(210)),v_values)
        
        action = np.argmax(prediction)
        #action = np.random.choice(range(0,self.action_space.n), p=[0.2,0.8,0.0])

        #print(prediction)
        #print(action)
        return action		
    def on_episode_start(self):
        self.update_count = 0
    def on_episode_end(self, reward, step_count):
        #if self.verbose:
        #	print('Episode end', reward, step_count, self.total_step_count)
        #self.sw.add([reward], self.total_step_count)
        #self.add_stat('reward', (self.total_step_count, reward))
        if len(self.losses)>0:
            x = np.array(self.losses)
            aver_loss = x.sum() / self.update_count
        else:
            aver_loss = 0
        if self.verbose:			
            print('Episode end', reward, step_count, self.total_step_count, aver_loss)
        self.update_count = 0
        self.sw.add([reward, aver_loss], self.total_step_count)
        self.losses = []
        self.add_stat('reward', (self.total_step_count, reward))
