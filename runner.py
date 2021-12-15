import numpy as np
import copy
from alg_parameters import *


class Runner(object):
    """Run multiple episode in multiprocessing environments."""

    def __init__(self, env, model):
        self.env = env
        self.model = model
        self.episode_rewards=np.zeros((N_ENVS,))
        self.env.reset()
        self.obs = env.get_obs()
        self.state = env.get_state()
        self.state = np.squeeze(self.state)
        self.dones = [False for _ in range(N_ENVS)]
        self.h_schedule_n=np.zeros((N_ENVS,N_AGENTS))
        self.schedule_n=np.zeros((N_ENVS,N_AGENTS))
        self.obs, self.state, _ = get_obs_state_with_schedule(self.obs, self.state, self.h_schedule_n, self.schedule_n)

    def run(self):
        mb_obs, mb_rewards, mb_values, mb_dones,ep_infos,mb_actions= [], [], [], [], [], []
        mb_sch_values,mb_state=[],[]
        mb_priority,mb_schedule_n=[],[]
        mb_ps=[]
        episode_rewrads_info=[]

        for _ in range(N_STEPS):
            mb_obs.append(self.obs.copy())
            mb_state.append(self.state.copy())
            valid_action = self.env.get_avail_actions()
            self.schedule_n, priority = self.model.act_model.get_schedule(self.obs) # correct
            actions, values,sch_values,ps = self.model.step(self.obs, self.state,self.schedule_n,priority,valid_action)
            mb_values.append(values)
            mb_sch_values.append(sch_values)
            mb_ps.append(ps)
            mb_dones.append(self.dones)
            mb_actions.append(actions)
            mb_priority.append(priority)
            mb_schedule_n.append(self.schedule_n)

            rewards, self.dones, infos = self.env.step(actions)
            self.episode_rewards+=rewards
            true_index = np.argwhere(self.dones == True)

            # Record information of episode when the episode is end
            if len(true_index) != 0:
                true_index = np.squeeze(true_index)
                self.h_schedule_n= copy.deepcopy(self.h_schedule_n)
                self.schedule_n = copy.deepcopy(self.schedule_n)
                self.h_schedule_n[true_index] = np.zeros((self.h_schedule_n[true_index].shape))
                self.schedule_n[true_index] = np.zeros((self.schedule_n[true_index].shape))
                episode_rewrads_info.append(np.nanmean(self.episode_rewards[true_index]))
                self.episode_rewards[true_index]=np.zeros(self.episode_rewards[true_index].shape)
                if true_index.shape==():
                    ep_infos.append(infos[true_index])
                else:
                    for item in true_index:
                        ep_infos.append(infos[item])
            self.obs = self.env.get_obs()
            self.state = self.env.get_state()
            self.state = np.squeeze(self.state)
            self.obs, self.state, self.h_schedule_n = get_obs_state_with_schedule(self.obs, self.state, self.h_schedule_n,
                                                                  self.schedule_n)
            mb_rewards.append(rewards)

        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_state = np.asarray(mb_state, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_sch_values = np.asarray(mb_values, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        mb_actions = np.asarray(mb_actions)  # [step,env,agent]
        mb_priority = np.asarray(mb_priority)
        mb_schedule_n = np.asarray(mb_schedule_n)
        mb_ps=np.asarray(mb_ps)
        # calculate advantages
        last_schedule_n, last_priority = self.model.act_model.get_schedule(self.obs)  # correct
        last_values,last_sch_value = self.model.value(self.state,last_priority)  # [env,agent]
        mb_advs = np.zeros_like(mb_rewards)
        mb_sch_advs = np.zeros_like(mb_rewards)
        lastgaelam,lastgaelam_sch = 0,0
        for t in reversed(range(N_STEPS)):
            if t == N_STEPS - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
                nextschvalues = last_sch_value
            else:
                nextnonterminal = 1.0 - mb_dones[t + 1]
                nextvalues = mb_values[t + 1]
                nextschvalues = mb_sch_values[t + 1]
            delta = mb_rewards[t] + GAMMA * nextvalues * nextnonterminal - mb_values[t]
            delta_sch = mb_rewards[t] + GAMMA * nextschvalues* nextnonterminal  - mb_sch_values[t]
            mb_advs[t] = lastgaelam = delta + GAMMA * LAM * nextnonterminal * lastgaelam
            mb_sch_advs[t] = lastgaelam_sch = delta_sch + GAMMA * LAM * nextnonterminal * lastgaelam_sch
        mb_returns = mb_advs + mb_values
        mb_sch_returns = mb_sch_advs + mb_sch_values
        return (*map(swap_flat, (mb_obs, mb_state,mb_returns, mb_values, mb_sch_returns,mb_sch_values,
                                 mb_actions,mb_ps,mb_priority,mb_schedule_n)), ep_infos,np.nanmean(episode_rewrads_info))



def swap_flat(arr):
    """
    swap and then flatten axes 0 and 1. """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def get_obs_state_with_schedule(obs_n_ws, state_n, h_schedule_n, schedule_n):
    h_schedule_n = h_schedule_n * 0.5 + schedule_n * 0.5
    obs_n_h = np.concatenate((obs_n_ws, h_schedule_n.reshape((N_ENVS,N_AGENTS, 1))), axis=-1)
    state = np.concatenate((state_n, h_schedule_n), axis=-1)
    return obs_n_h, state, h_schedule_n
