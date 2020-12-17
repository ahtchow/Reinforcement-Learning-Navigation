import numpy as np
import random
from collections import namedtuple, deque

version = 1.0
EXP_FIELD_NAME = ["state", "action", "reward", "next_state", "done"]

class ReplayBuffer:
    """ 
    ReplayBuffer (Data Structure): Fixed-size buffer to store experience tuples.
    Experience (Data): NamedTuple with structure of
                       ["state", "action", "reward", "next_state", "done"]

    """

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """
        Initialize a ReplayBuffer object.
        
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            device (string): compute resource
        
        """

        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = seed
        self.device = device

        # Memories will be stored on a double-ended queue (deque)
        self.memory = deque(maxlen=self.buffer_size)
        
        # Experiences will be a named tupile structure
        
        self.experience = namedtuple("Experience", field_names=EXP_FIELD_NAME)
        
    
    def add(self, state, action, reward, next_state, done):
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
        
        exp = self.experience(state, action, reward, next_state, done)
        self.memory.append(exp)

    
    def sample(self)
        """ 
        
        Randomly sample an experience from the memory. 
        
        return: experience (tuple)
        
        """

        # Sample an experience with length k from list of memories
        experiences = random.sample(self.memory, k=self.batch_size)

        # For each item in the tuple, stack vertically and convert to GPU torch tensor
        states = np.vstack([e.state for e in experiences if e is not None])
        states = torch.from_numpy(states).float().to(self.device) # (float)

        actions = np.vstack([e.action for e in experiences if e is not None])
        actions = torch.from_numpy(action).long().to(self.device) # (long)

        rewards = np.vstack([e.reward for e in experiences if e is not None])
        rewards = torch.from_numpy(rewards).float().to(self.device) # (float)

        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        next_states = torch.from_numpy(next_states).float().to(self.device) # float

        dones = np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8) # Make bool an int
        dones = torch.from_numpy(dones).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """
        
        Return the current size of internal memory.
        
        return: length (int)
        
        """
        return len(self.memory)