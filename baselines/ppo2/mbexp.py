import argparse
import gym
import numpy as np
import time
import math
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, RandomSampler, BatchSampler
from torch.distributions import Categorical

#########################
# Reward, done function #
#########################
def gym_pendulum(s, a, sp):
    def angle_normalize(x):
        return (((x+np.pi) % (2*np.pi)) - np.pi)

    max_speed = 8
    max_torque = 2.

    #th, thdot = sp # th := theta
    theta_cos, theta_sin, theta_dot = sp
    theta_cos = np.clip(theta_cos, -1.0, 1.0)
    th = np.arccos(theta_cos)
    thdot = theta_dot

    g = 10.
    m = 1.
    l = 1.
    dt = .05

    u = np.clip(a, -max_torque, max_torque)
    costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

    return -costs, float(False)

def gym_cartpole_reward_done_func(s, a, sp):
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = (masspole + masscart)
    length = 0.5 # actually half the pole's length
    polemass_length = (masspole * length)
    force_mag = 10.0
    tau = 0.02  # seconds between state updates

    # Angle at which to fail the episode
    theta_threshold_radians = 12 * 2 * math.pi / 360
    x_threshold = 2.4
    
    state = sp
    x, _, theta, _ = state
    
    done =  x < -x_threshold \
            or x > x_threshold \
            or theta < -theta_threshold_radians \
            or theta > theta_threshold_radians
    done = bool(done)

    if not done:
        reward = 1.0
    else:
        reward = 0.0
    return reward, float(done)

def get_reward_done_func(env_id):
    dicts = {'CartPole-v0': gym_cartpole_reward_done_func,
            'Pendulum-v0': gym_pendulum}
    return dicts[env_id]
#########################
#########################

#########################
# Forward dynamic model #
#########################
class ForwardDynamicModel(nn.Module):
    def __init__(self, ob_space, ac_space, lr=1e-3):
        super(ForwardDynamicModel, self).__init__()
        self.ob_space = ob_space
        self.ac_space = ac_space
    
        n_obs = self.ob_space.shape[-1]
        if isinstance(ac_space, gym.spaces.Discrete):
            n_act = self.ac_space.n
        else:
            n_act = self.ac_space.shape[-1]

        self.w = nn.Sequential(
            nn.Linear(n_obs + n_act, 128),
            nn.ReLU(),
            nn.Linear(128, n_obs)
        )

        self.loss = nn.MSELoss(True, True)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.n_obs = n_obs
        self.n_act = n_act

    def forward(self, s, a, add_diff=False):
        if isinstance(self.ac_space, gym.spaces.Discrete):
            a = torch.cat([torch.eye(self.n_act)[a_.long()].unsqueeze(0) for a_ in a]).float()
        else:
            print(a.size())
            a = a.view((-1,) + self.ac_space.shape)
        x = torch.cat([s, a], dim=1)
        y = self.w(x)
        if add_diff:
            return s + y
        return y

    def train(self, s, a, sp):
        self.optimizer.zero_grad()
        pred = self.forward(s, a)
        targ = sp - s
        loss = self.loss(pred, targ)
        loss.backward()
        self.optimizer.step()
        return loss

def train_forward_dynamic_model(fwd_model, transitions, n_epochs, batch_size):
    s_train = np.array([e[0] for e in transitions])
    a_train = np.array([e[1] for e in transitions])
    sp_train = np.array([e[2] for e in transitions])
    s_train_pth = torch.from_numpy(s_train).float()
    a_train_pth = torch.from_numpy(a_train).float()
    sp_train_pth = torch.from_numpy(sp_train).float()

    dataset = TensorDataset(s_train_pth, a_train_pth, sp_train_pth)
    sampler = BatchSampler(RandomSampler(dataset), batch_size, False)
    for batch in sampler:
        loss = fwd_model.train(s_train_pth[batch], a_train_pth[batch], sp_train_pth[batch])
        print('Loss', loss)

def build_forward_dynamic_model(env):
    fwd_model = ForwardDynamicModel(env.observation_space, env.action_space)
    transitions = []
    s = env.reset()
    while len(transitions) < 5000:
        a = env.action_space.sample()
        sp, r, d, _ = env.step(a[np.newaxis, ...])
        transitions.append((s.copy()[0], a, sp.copy()[0]))
        s = sp
        if d:
            s = env.reset()
    train_forward_dynamic_model(fwd_model, transitions, 50, 64)
    print('Finish forward dynamic pretraining.')
    return fwd_model

def lookahead(pi, fwd_model, reward_done_func, state, horizon, num_samples):
    state = torch.from_numpy(state).float()
    state_shape = list(state.size())
    state_dim = state_shape[-1]
    all_reward_mask = np.ones(num_samples)
    all_total_reward = np.zeros(num_samples)
    all_state = state.repeat(1, num_samples).view(num_samples, state_dim)
    all_history_action = []
    all_history_value = []
    all_history_neglogprob = []
    for h in range(horizon):
        # Get the action, value, and action neglogprob from TF policy
        all_action_and_value_and_states_and_neglogprob = [pi(s.data.numpy()) for s in all_state] # N x (act_dim, 1)
        # Convert the action to torch.Tensor for fwd_model
        all_action = torch.FloatTensor([al[0] for al in all_action_and_value_and_states_and_neglogprob])
        # Extract value and neglogprob
        all_value = [al[1] for al in all_action_and_value_and_states_and_neglogprob] # N x 1
        all_neglogprob = [al[2] for al in all_action_and_value_and_states_and_neglogprob] # N x 1
        # Forward simulate
        all_state_next = fwd_model.forward(all_state, all_action, add_diff=True) # N x state_dim
        # Evaluate the next state
        all_reward_done = [reward_done_func(s.data.numpy(), a.data.numpy(), sp.data.numpy()) for s, a, sp in zip(all_state, all_action, all_state_next)] # N x (1, 1)
        all_reward = np.array([rd[0] for rd in all_reward_done]) # N x 1
        all_done = np.array([rd[1] for rd in all_reward_done]) # N x 1
        all_total_reward += (all_reward * all_reward_mask) # N x 1
        all_history_action.append(all_action.data.numpy()[np.newaxis, ...]) # H x N x act_dim
        all_history_value.append(all_value)
        all_history_neglogprob.append(all_neglogprob)
        all_reward_mask = np.ones_like(all_done) - all_done # N x 1 
        all_state = all_state_next # N x state_dim
    all_history_action = np.concatenate(all_history_action, axis=0) # H X N x act_dim
    best_idx = np.argmax(all_total_reward)
    best_action = all_history_action[0, best_idx, ...]
    best_value = all_history_value[0, best_idx]
    best_neglogprob = all_history_neglogprob[0][best_idx]
    #print('Est. rew', all_total_reward.max(), all_total_reward.min(), all_total_reward.mean())
    return best_action, best_value, best_neglogprob

class MBEXP(object):
    def __init__(self, env, env_id, pi):
        self.env = env
        self.env_id = env_id
        self.pi = pi
        self.ob_space = env.observation_space
        self.ac_space = env.action_space
        self.reward_done_func = get_reward_done_func(env_id)
        self.fwd_model = build_forward_dynamic_model(self.env)
    
    def step(self, observation, recurrent_state, done, num_samples=5, horizon=10):
        action, value, neglogprob = lookahead(self.pi, self.fwd_model, self.reward_done_func, 
                        observation, horizon, num_samples)
        return action[np.newaxis, ...], value[np.newaxis, ...], None, neglogprob[np.newaxis, ...]

#########################
#########################