import numpy as np
import random
from collections import deque

from model import Critic, tdActor

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
class TD3_agent:
    def __init__(self, states = 48, actions = 4, gamma = 0.95, lr = 0.0001, tau = 0.01, action_noise = 0.2, policy_smooth_noise = 0.3, noise_clip = 0.5, policy_delay = 4, batch_size = 64, agent_ID = 0):
        
        ## Critics 
        self.qnet1 = Critic(states, actions).to(device)
        self.qtarget1 = Critic(states,actions).to(device)
        self.q1_optimizer = optim.Adam(self.qnet1.parameters(), lr = lr)
        self.soft_update(1, self.qtarget1, self.qnet1)
        
        self.qnet2 = Critic(states, actions).to(device)
        self.qtarget2 = Critic(states,actions).to(device)
        self.q2_optimizer = optim.Adam(self.qnet2.parameters(), lr = lr)
        self.soft_update(1, self.qtarget2, self.qnet2)
        
        ## policies -> input half of the all states
        self.policy = tdActor(int(states/2), int(actions/2)).to(device)
        self.policy_target = tdActor(int(states/2), int(actions/2)).to(device)
        self.soft_update(1, self.policy_target, self.policy)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr = lr)

        self.q_criteria = torch.nn.MSELoss()
                
        ## uncorrelated gaussian noise 
        self.action_noise = action_noise
        self.policy_smooth_noise = policy_smooth_noise
        
        ## update parameters
        self.agent_ID = agent_ID
        self.action_size = actions
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.steps = 0
       
        self.noise_clip = noise_clip
        self.batch_size = batch_size
        
        ## noise samplers
        self.sampler =  Normal(torch.zeros(batch_size, int(actions/2)), self.policy_smooth_noise * torch.ones(batch_size, int(actions/2)))
        self.action_sampler = Normal(torch.zeros(int(actions/2)), self.action_noise * torch.ones(int(actions/2)))
    
    def act(self, state, noise  =True):
        if type(state) == np.ndarray:
            state = torch.from_numpy(state).float().to(device)
        with torch.no_grad():
            actions = self.policy(state)
        if noise:
            actions += self.action_sampler.sample().to(device)
            actions = torch.clamp(actions, -1, 1)
        return actions.view(1, -1)
        
    def step(self, ready = False, experiences = None):

        # If enough samples are available in memory, get random subset and learn
        if ready:
            ## time to gradient descent
            self.learn(experiences)    
            self.steps += 1

    def learn(self, experiences):
                
        states, actions, rewards, next_states, dones = experiences
 
        local_states = states[:,self.agent_ID,:]
        local_actions = actions[:,self.agent_ID,:]
        local_rewards = rewards[:,self.agent_ID,:]
        local_next_states = next_states[:,self.agent_ID,:]
        local_dones = dones[:,self.agent_ID,:]
 
        
        with torch.no_grad():
            sampled_actions = self.policy_target(local_next_states)
            sampled_actions =torch.clamp(sampled_actions+torch.clamp(self.sampler.sample().to(device), -self.noise_clip, self.noise_clip), -1, 1) 
            sampled_actions_collab = self.collaborator_policy(states[:,1-self.agent_ID,:])
            sampled_actions_collab = torch.clamp(sampled_actions_collab + torch.clamp(self.sampler.sample().to(device), -self.noise_clip, self.noise_clip), -1, 1)
            
            if self.agent_ID == 0:
                actions_cat = torch.cat([sampled_actions, sampled_actions_collab], -1)
            else:
                actions_cat = torch.cat([sampled_actions_collab, sampled_actions], -1)
            
            
            q_target_1 = self.qtarget1(next_states.view(self.batch_size, -1), actions_cat)
            q_target_2 = self.qtarget2(next_states.view(self.batch_size, -1), actions_cat)
        
        y = self.gamma * torch.min(q_target_1, q_target_2) * (1-local_dones) + local_rewards
        
        ## -------------- update critics --------------- ##
        q_expected_1 = self.qnet1(states.view(self.batch_size, -1), actions.view(self.batch_size, -1))
        q_expected_2 = self.qnet2(states.view(self.batch_size, -1), actions.view(self.batch_size, -1))
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        
        q1_loss = self.q_criteria(q_expected_1, y)
        q2_loss = self.q_criteria(q_expected_2, y)
        
        q1_loss.backward()
        q2_loss.backward()
        
        ## gradient clipping
        torch.nn.utils.clip_grad_norm_(self.qnet1.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(self.qnet2.parameters(), 1)
        
        self.q1_optimizer.step()
        self.q2_optimizer.step()  
        
        
        ## --------------- update policy ----------------##
        if self.steps % self.policy_delay == 0:
 
            
            self.policy_optimizer.zero_grad()
            
            policy_actions = self.policy(local_states) 
            
            ## instead of using collaborator's policy, use sampled actions
            if self.agent_ID == 0:
#                 print(policy_actions.size(), actions[:,1-self.agent_ID, :].size())
                policy_actions = torch.cat([policy_actions, actions[:,1-self.agent_ID, :]], -1)
            else:
                policy_actions = torch.cat([actions[:,1-self.agent_ID, :], policy_actions], -1)
            
            p_loss = -self.qnet1(states.view(self.batch_size, -1), policy_actions).mean()
            
            p_loss.backward()
            
#             torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 3)
            self.policy_optimizer.step()
            
            # soft update
            self.soft_update(self.tau, self.qtarget1, self.qnet1)
            self.soft_update(self.tau, self.qtarget2, self.qnet2)     
            self.soft_update(self.tau, self.policy_target, self.policy)
            
    @staticmethod
    def soft_update(tau, target_net, local_net):
        for target_param, local_param in zip(target_net.parameters(),  local_net.parameters()):
            target_param.data.copy_( tau * local_param.data + (1-tau) * target_param.data)  
    
    def noisy_action(self, policy, states):
        target_actions = policy(states)
        target_actions =torch.clamp(target_actions+torch.clamp(self.sampler.sample().to(device), -self.noise_clip, self.noise_clip), -1, 1)
        return target_actions
        

class MA_TD3:
    def __init__(self, states = 48, actions = 4, gamma = 0.995, lr = 0.005, tau = 0.005, action_noise = 0.15, policy_smooth_noise = 0.2, noise_clip = 0.4, policy_delay = 2, batch_size = 64, seed = 42, buffer_size = int(1e5) ):
        
        ## set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        ## MAgents
        self.agents = [TD3_agent(states = states, actions = actions, action_noise = action_noise, 
                                 policy_smooth_noise =policy_smooth_noise, noise_clip = noise_clip, agent_ID = 0), 
                       TD3_agent(states = states, actions = actions, action_noise = action_noise, 
                                 policy_smooth_noise =policy_smooth_noise, noise_clip = noise_clip, agent_ID = 1)]
        ## update parameters
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.steps = 0
        self.batch_size = batch_size
        self.replay = EReplay(buffer_size, actions, batch_size, seed)
 
        self.noise_clip = noise_clip
        
    def act(self, state, noise=True):
        
        acts = []
        for idx, s in enumerate(state):
            acts.append(self.agents[idx].act(s, noise))
        ## return actions in a numpy array (2,2)
        return torch.cat(acts, 0).cpu().detach().numpy()
    
    def step(self, states, actions, rewards, next_states, dones):
        # Save experience in replay memory
        self.replay.add(states, actions, rewards, next_states, dones)

        # If enough samples are available in memory, get random subset and learn
        if self.replay.ready():
            ## time to gradient descent
            
            experiences = self.replay.sample()
            self.learn(experiences, agentID = 0)
            
            experiences = self.replay.sample()
            self.learn(experiences, agentID = 1) 
            self.steps += 1
    
    def learn(self, experiences, agentID):
                
        states, actions, rewards, next_states, dones = experiences
 
        local_states = states[:,agentID,:]
        local_actions = actions[:,agentID,:]
        local_rewards = rewards[:,agentID,:]
        local_next_states = next_states[:,agentID,:]
        local_dones = dones[:,agentID,:]
        
        current_agent = self.agents[agentID]
        other_agent = self.agents[1-agentID]
 
        
        with torch.no_grad():
            current_target_actions = current_agent.noisy_action(current_agent.policy_target, local_next_states)
            other_target_actions   = other_agent.noisy_action(other_agent.policy_target, next_states[:, 1-agentID,:])
            
            if agentID == 0:
                actions_cat = torch.cat([current_target_actions, other_target_actions], -1)
            else:
                actions_cat = torch.cat([other_target_actions, current_target_actions], -1)
            
            
            q_target_1 = current_agent.qtarget1(next_states.view(self.batch_size, -1), actions_cat)
            q_target_2 = current_agent.qtarget2(next_states.view(self.batch_size, -1), actions_cat)
        
        y = self.gamma * torch.min(q_target_1, q_target_2) * (1-local_dones) + local_rewards
        
        ## -------------- update critics --------------- ##
        q_expected_1 = current_agent.qnet1(states.view(self.batch_size, -1), actions.view(self.batch_size, -1))
        q_expected_2 = current_agent.qnet2(states.view(self.batch_size, -1), actions.view(self.batch_size, -1))
        current_agent.q1_optimizer.zero_grad()
        current_agent.q2_optimizer.zero_grad()
        
        q1_loss = current_agent.q_criteria(q_expected_1, y)
        q2_loss = current_agent.q_criteria(q_expected_2, y)
        
        q1_loss.backward()
        q2_loss.backward()
        
        ## gradient clipping
        torch.nn.utils.clip_grad_norm_(current_agent.qnet1.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(current_agent.qnet2.parameters(), 1)
        
        current_agent.q1_optimizer.step()
        current_agent.q2_optimizer.step()  
        
        
        ## --------------- update policy ----------------##
        if self.steps % self.policy_delay == 0:

            current_agent.policy_optimizer.zero_grad()
            
            policy_actions = current_agent.policy(local_states) 
            
            ## instead of using collaborator's policy, use sampled actions
            if agentID == 0:
#                 print(policy_actions.size(), actions[:,1-self.agent_ID, :].size())
                policy_actions = torch.cat([policy_actions, actions[:,1-agentID, :]], -1)
            else:
                policy_actions = torch.cat([actions[:,1-agentID,:], policy_actions], -1)
            
            p_loss = -current_agent.qnet1(states.view(self.batch_size, -1), policy_actions).mean()
            
            p_loss.backward()
            
#             torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 3)
            current_agent.policy_optimizer.step()
            
            # soft update
            self.soft_update(self.tau, current_agent.qtarget1, current_agent.qnet1)
            self.soft_update(self.tau, current_agent.qtarget2, current_agent.qnet2)     
            self.soft_update(self.tau, current_agent.policy_target, current_agent.policy)
            
    @staticmethod
    def soft_update(tau, target_net, local_net):
        for target_param, local_param in zip(target_net.parameters(),  local_net.parameters()):
            target_param.data.copy_( tau * local_param.data + (1-tau) * target_param.data)
        
class TD3_agent_original:
    def __init__(self, states = 24, actions = 2, gamma = 0.99, lr = 3e-4, tau = 0.005, policy_smooth_noise = 0.2, noise_clip = 0.3, policy_delay = 2, batch_size = 64, seed = 10, start_steps = 10000):
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        ## Critics 
        self.qnet = Critic(states, actions).to(device)
        self.qnet_target = Critic(states,actions).to(device)
        self.q_optimizer = optim.Adam(self.qnet.parameters(), lr = lr)
        self.soft_update(1, self.qnet_target, self.qnet)
        
        ## policies
        self.policy = tdActor(states, actions).to(device)
        self.policy_target = tdActor(states, actions).to(device)
        self.soft_update(1, self.policy_target, self.policy)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr = lr)
        
        self.q_criteria = torch.nn.MSELoss()
        
        ## Replay buffer
        self.replay = EReplay(int(1e5), batch_size = batch_size)
        
        ## uncorrelated gaussian noise 
        self.policy_smooth_noise = policy_smooth_noise
        
        ## update parameters
        self.action_size = actions
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.steps = 0
        self.noise_clip = noise_clip
        self.start_steps = start_steps
        
    def act(self, state):
        if type(state) == np.ndarray:
            state = torch.from_numpy(state).float().to(device)
        act = []
        with torch.no_grad(): 
            for s in state:
                actions = self.policy(s).view(1, -1)
                act.append(actions)
        return torch.cat(act, 0).cpu().detach().numpy()
        
    def step(self, states, actions, rewards, next_states, dones):
        # Save experience in replay memory
        for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
            self.replay.add(s, a, r, ns, d)

        # If enough samples are available in memory, get random subset and learn
        if self.replay.ready(self.start_steps):
            ## time to gradient descent
      
            experiences = self.replay.sample()
            self.learn(experiences )    
            self.steps += 1

    def learn(self, experiences ):
        
        states, actions, rewards, next_states, dones = experiences
        
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_smooth_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.policy_target(next_states) + noise).clamp(-1, 1)
            q1_target, q2_target = self.qnet_target(next_states, next_actions)
            q_target = torch.min(q1_target, q2_target)
            q_target = rewards + self.gamma * (1-dones) * q_target
        
        ## -------------- update critics --------------- ##
        q1_exp, q2_exp = self.qnet(states, actions)
        q_loss = self.q_criteria(q1_exp, q_target) + self.q_criteria(q2_exp, q_target)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()        
        
        ## --------------- update policy ----------------##
        if self.steps % self.policy_delay == 0:
            p_loss = -self.qnet.Q1(states, self.policy(states)).mean()
            self.policy_optimizer.zero_grad()
            p_loss.backward()
            self.policy_optimizer.step()            
 
            # soft update
            self.soft_update(self.tau, self.qnet_target, self.qnet)
            self.soft_update(self.tau, self.policy_target, self.policy)
            
    @staticmethod
    def soft_update(tau, target_net, local_net):
        for target_param, local_param in zip(target_net.parameters(),  local_net.parameters()):
            target_param.data.copy_( tau * local_param.data + (1-tau) * target_param.data)         

            
## Normal experience replay, to be upgraded to be priority experience replay
class EReplay:
    def __init__(self, size = int(1e5), action_size = 4, batch_size = 128, seed = 42):
        self.memory = deque(maxlen = size)
        self.batch_size = batch_size
        self.action_size = action_size 
        random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self):
        experience = random.sample(self.memory, k = self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experience)
        
        states = torch.from_numpy(np.asarray(states)).float().to(device) 
        try:
            actions = torch.from_numpy(np.asarray(actions)).float().to(device) 
            rewards = torch.from_numpy(np.asarray(rewards)).float().to(device)
            next_states = torch.from_numpy(np.asarray(next_states)).float().to(device) 
            dones = torch.from_numpy(np.asarray(dones)).float().to(device)
        except:
            print(actions[0].shape)
            print('Error raised')
  
        return (states, actions, rewards, next_states, dones)
        
    def ready(self, start_steps):
        return len(self.memory) >= start_steps
    
    
    def __len__(self):
        return len(self.memory)


        
        