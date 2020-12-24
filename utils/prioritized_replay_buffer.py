import numpy as np
import torch
import random
from collections import namedtuple, deque

version = 1.0
EXP_FIELD_NAME = ["state", "action", "reward", "next_state", "done", "prob"]

class PrioritizedReplayBuffer:
    """ 
    ReplayBuffer (Data Structure): Fixed-size buffer to store experience tuples.
    Experience (Data): NamedTuple with structure of
                       ["state", "action", "reward", "next_state", "done"]

    """

    def __init__(self, 
                 action_size, 
                 buffer_size, 
                 batch_size, 
                 seed, 
                 device,
                 prioritized_params):
        """
        Initialize a ReplayBuffer object.
        
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            device (string): compute resource
            prioritized_params (dict): contains the constants for hyperparameters 
              α (int): α = 0 corresponding to the uniform case, α = 1 to using priorities
              b (int): b is the bias used to scale the change in weights for importance-sampling weight
        
        """
        
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        self.a = prioritized_params['a']
        self.b = prioritized_params['b']
        self.b_inc_rate = prioritized_params['b_inc_rate']
        self.e = prioritized_params['e']
        self.b_max = 1.0
        self.max_prob = 1.0
        
        # Memories will be stored on a double-ended queue (deque)
        self.memory = deque(maxlen=self.buffer_size)
        # Now we have to keep track of priorities as well
        self.priorities = deque(maxlen=self.buffer_size)
        
        # Experiences will be a named tupile structure
        self.experience = namedtuple("Experience", field_names=EXP_FIELD_NAME)
        
    
    
    def get_td_errors(self, state, action, reward, next_state, done, models):
        
        """ Calculate td error."""
        models['local'].eval() # Evaluation Mode
        with torch.no_grad(): # No Gradient Descent
            
            # Make 64 copies to fit into model
            states = []
            rewards = []
            next_states = []
            dones = []
            
            for i in range(self.batch_size):
                states.append(state)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
                
            states = torch.from_numpy(np.vstack(states)).float().to(self.device)
            q_expected = models['local'].foward(states)

            next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
            q_prime = models['target'].foward(next_states)

            rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
            dones = torch.from_numpy(np.vstack(dones)).float().to(self.device)
            not_done_bool = (1 - dones) # If done, no need to include next return
            td_target =  rewards + (models['GAMMA'] * q_prime) * not_done_bool

            td_error = td_target - q_prime
            
        models['local'].train()
        
        return abs(td_error.detach()[0][action])
 
    
    def add(self, state, action, reward, next_state, done, models):
        """ 
        
        Append an experience to the memory. 
        
        Params
        ======
            state (continious): current state before action
            action (discrete): action take for given state
            reward (int): reward recieved after performing action
            next_state (int): state achieved at timestep t+1
            done (bool): episode completed on this timestep
        
        """
        # Create an experience
        
        #Calculate the td_error
        td_error = self.get_td_errors(state, action, reward, next_state, done, models)
        td_error += self.e
        exp = self.experience(state, action, reward, next_state, done, td_error)
        self.memory.append(exp)
        self.priorities.append(td_error)

        
    def sample(self):
        """ 
        
        Randomly sample an experience from the memory. 
        
        return: experience (tuple)
        
        """
        
        ########################################
        #  Prioritized Sampling Modifications  #
        ########################################
        
        # Calculate the Sampling priorities
        sampling_prob = np.array(self.priorities)
        sampling_prob = sampling_prob ** self.a / sum(sampling_prob ** self.a)
        sample_idxs = np.random.choice(np.arange(len(self.memory)), size=self.batch_size, p=sampling_prob)
        experiences = []
        update_factors = []
        for i in sample_idxs:
            experiences.append(self.memory[i])
            update_factors.append( ((1 / self.buffer_size ) * (1 / sampling_prob[i] ) )** self.b ) 

        # For each item in the tuple, stack vertically and convert to GPU torch tensor
        states = np.vstack([e.state for e in experiences if e is not None])
        states = torch.from_numpy(states).float().to(self.device) # (float)

        actions = np.vstack([e.action for e in experiences if e is not None])
        actions = torch.from_numpy(actions).long().to(self.device) # (long)

        rewards = np.vstack([e.reward for e in experiences if e is not None])
        rewards = torch.from_numpy(rewards).float().to(self.device) # (float)

        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        next_states = torch.from_numpy(next_states).float().to(self.device) # float

        dones = np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8) # Make bool an int
        dones = torch.from_numpy(dones).float().to(self.device)

        update_factors = np.vstack(update_factors)
        update_factors = torch.from_numpy(update_factors).float().to(self.device)
        
        # Update b
        self.b = min(self.b * self.b_inc_rate, self.b_max)
        
        return (states, actions, rewards, next_states, dones, update_factors)

    def __len__(self):
        """
        
        Return the current size of internal memory.
        
        return: length (int)
        
        """
        return len(self.memory)