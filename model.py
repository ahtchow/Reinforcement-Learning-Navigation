import torch
import torch.nn as nn

class QNetwork(nn.Module):
    
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        
        """
        
        Actor (Policy) Model
        
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer

        """
        
        # Inherit nn.Module as subclass
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        self.state_size = state_size
        
         # Define layer parameters
        self.fully_connected_1 = nn.Linear(state_size, fc1_units)
        self.fully_connected_2 = nn.Linear(fc1_units, fc2_units)
        self.fully_connected_3 = nn.Linear(fc2_units, action_size)
        
    def foward(self, state):
        """
        Foward Pass that maps state -> action values.
        
        """
        X = nn.functional.relu(self.fully_connected_1(state))
        X = nn.functional.relu(self.fully_connected_2(X))
        return self.fully_connected_3(X)

    
class DuelingQNetwork(nn.Module):
    
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64, fc3_units=32):

                
        """
        
        Actor (Policy) Model
        
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            fc2_units (int): Number of nodes in thirs hidden layer

        """
        
        # Inherit nn.Module as subclass
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.feauture_layer = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU()
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(fc2_units, fc3_units),
            nn.ReLU(),
            nn.Linear(fc3_units, action_size),
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(fc2_units, fc3_units),
            nn.ReLU(),
            nn.Linear(fc3_units, action_size),
        )
        

        
    def foward(self, state):
        """
        Foward Pass that maps state -> action values.
        
        """
        # Foward Pass
        features = self.feauture_layer(state)
        
        # Value Stream - State Values
        values = self.value_stream(features)
        # Advantage Stream - Advantage Values
        advantages  = self.advantage_stream(features)
        
        # Q Estimate = V(state) + (A(state) - mean(A(state))
        qvals = values + (advantages - advantages.mean())
        
        return qvals