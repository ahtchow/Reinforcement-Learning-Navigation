import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import random

from utils.replay_buffer import ReplayBuffer
from utils.prioritized_replay_buffer import PrioritizedReplayBuffer
from model import QNetwork, DuelingQNetwork

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor    
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, 
                 state_size, 
                 action_size, 
                 seed, 
                 double_DQN=False, 
                 prioritized_replay=False,
                 dueling_networks=False):
        
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            double_DQN (bool) : use double DQN
            prioritized_replay (bool): used prioritized_replay
            

        """

        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.tau = TAU
        self.double_DQN = double_DQN
        self.prioritized_replay = prioritized_replay
        self.dueling_networks = dueling_networks
        
        if self.dueling_networks:
            # Q-Networks - Local, Target Neural Nets
            self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target.eval()
            
        else:
            # Q-Networks - Local, Target Neural Nets
            self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target.eval()
        
        # Use optimizer to update the "local" neural net
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        
        if self.prioritized_replay:
            prioritized_params = {'a': 0.6, 'b': 0.4, 'b_inc_rate': 1.001, 'e': 0.01}
            self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, device, prioritized_params)
        else:
            # Replay memory
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, device)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0


    def act(self, state, eps=0.):
        """
        
        Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection


        eval() -notify all your layers that you are in eval mode, that way, 
        batchnorm or dropout layers will work in eval mode instead of training 
        mode.
        
        no_grad() - impacts the autograd engine and deactivate it. It will reduce 
        memory usage and speed up computations but you won’t be able to backprop.

        
        """

        # Process state to a GPU tensor, increases dimension on x-axis (dim=0)
        state = torch.from_numpy(state).float()
        state = state.unsqueeze(0).to(device) 

        self.qnetwork_local.eval() # Evaluation Mode
        with torch.no_grad(): # No Gradient Descent
            # Returns vector of action values
            action_values = self.qnetwork_local.foward(state)

        # Epsilon-greedy action selection
        rand_from_0_to_1 = random.random()

        if rand_from_0_to_1 > eps:
            greedy_action_to_cpu = action_values.cpu().data.numpy()
            action = np.argmax(greedy_action_to_cpu) # get max value index
        else:
            action = random.choice(np.arange(self.action_size))

        self.qnetwork_local.train() # Back to train mode
        return int(action.item())


    def step(self, state, action, reward, next_state, done):
        """ 
        
        Process a step from time step t to t+1 by updating agent models. 
        
        Params
        ======
            state (continious): current state before action
            action (discrete): action take for given state
            reward (int): reward recieved after performing action
            next_state (int): state achieved at timestep t+1
            done (bool): episode completed on this timestep
        
        """
        if self.prioritized_replay:
            models = {'local': self.qnetwork_local, 'target': self.qnetwork_target, 'GAMMA': GAMMA}
            self.memory.add(state, action, reward, next_state, done, models)            
        else:   
            self.memory.add(state, action, reward, next_state, done)
        
        # Increase counter until we are ready to take an update step
        self.t_step = (self.t_step + 1) % UPDATE_EVERY

        if self.t_step == 0:
            # If enough samples are available in memory, get random subset
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
 
        
    def learn(self, experiences, gamma):
        """
        
        Update value parameters using given batch of experience tuples.
        
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor        
        
        """  
       
        # Note: These tensors make up a batch of 64 experiences
        
        # Unpack experience batch
        if self.prioritized_replay:
            states, actions, rewards, next_states, dones, update_factors = experiences
        else:
            states, actions, rewards, next_states, dones = experiences
        
        ###########################
        # Double DQN Modification #
        ###########################
        
        if self.double_DQN:
            
            # Get max predicted Q values (for next states) from local model
            q_prime = self.qnetwork_local.foward(next_states)
            # For the batch, we want to store the most greedy action for Double DQN Networks
            greedy_action_next = q_prime.max(dim=1,keepdim=True)[1]
            # Choose the reward from action that gives max return
            q_prime = q_prime.detach().max(1)[0].unsqueeze(1)
            
            # For DDQN, we must run the perform a sanity check using both nets, 
            # while still taking the original greedy actions
            DDQN_q_prime = self.qnetwork_target.foward(next_states)
            DDQN_q_prime = DDQN_q_prime.gather(1, greedy_action_next)
           
            not_done_bool = (1 - dones) # If done, no need to include next return
            td_target = rewards + (gamma * DDQN_q_prime) * not_done_bool
            
        
        #################
        # Standard DQN  #
        #################         
        
        else:
            
            # Get max predicted Q values (for next states) from target model
            q_prime = self.qnetwork_target.foward(next_states)
            # Choose the reward from action that gives max return
            q_prime = q_prime.detach().max(1)[0].unsqueeze(1)

            not_done_bool = (1 - dones) # If done, no need to include next return
            td_target = rewards + (gamma * q_prime) * not_done_bool
            

        
        # This is the model we will update
        q_expected = self.qnetwork_local.foward(states)
        # Gathers the expected values for each action
        q_expected = q_expected.gather(1, actions)       

        if self.prioritized_replay:
            q_expected *= update_factors
            td_target *= update_factors
        
        # Compute the loss, minimize the loss
        loss = F.mse_loss(td_target, q_expected)
        self.optimizer.zero_grad() # reset gradient
        loss.backward() # Calculate the gradient
        self.optimizer.step()  # Update weights

        # Update the target model parameters (Soft Update)
        # Soft Update: Factor in local parameter changes by a factor of TAU
        # Rather than update for every C steps, this helps inch closer to local parameters

        for target_param, local_param in zip(self.qnetwork_target.parameters(), 
                                             self.qnetwork_local.parameters()):

            upd_wghts = ((1.0 - self.tau) * target_param.data) + (self.tau * local_param.data)
            target_param.data.copy_(upd_wghts)
        
