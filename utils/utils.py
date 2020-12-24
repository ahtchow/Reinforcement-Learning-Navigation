import matplotlib.pyplot as plt
import numpy as np
from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import torch

def plot_scores(scores, scores_means, execution_info):
    
    ## Plot the scores
    fig = plt.figure(figsize=(20,10))
    
    for key in execution_info:
        print(f'{key}: {execution_info[key]}')

    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.plot(np.arange(len(scores_means)), scores_means)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend(('Score', 'Mean'), fontsize='xx-large')

    plt.show()

    
def learn_to_navigate_for_yellow_bananas(agent,
                                         env,
                                         brain_name,
                                         save_name='unidentified_params.pth',
                                         n_episodes=2000,
                                         max_t=1000,
                                         eps_start=1.0, 
                                         eps_end=0.1, 
                                         eps_decay=0.995,
                                         print_stats_every=100,
                                         win_condition=13.0):
    """
    
    Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    
    
    """

    scores = []
    scores_mean = []
    scores_window = deque(maxlen=100) # Score last 100 scores
    
    epsilon = eps_start # Define initial epsilon
    
    print("Beginning Training....")
    
    ####################
    # For each episode #
    ####################
    for i_episode in range(1, n_episodes + 1):
        
        # Define the an episode
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        
        score = 0
        
        #####################
        # For each timestep #
        #####################       
        
        for t in range(max_t):
            
            # Use agent model to predict next action
            action = agent.act(state, epsilon)
            
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished            
            
            agent.step(state, action, reward, next_state, done)

            # Store results
            state = next_state
            score += reward
            
            if done:
                break
                
        epsilon = max(eps_end, eps_decay*epsilon) # update/decrease epsilon
        
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        scores_mean.append(np.mean(scores_window)) # save means of last 100 trials
        
        ##################
        # Printing Stats #
        ##################
        
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        
        # Print on print_every condition
        if i_episode % print_stats_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            
        # Winning condition + save model parameters    
        if np.mean(scores_window)>= win_condition:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), save_name)
            break
    
    execution_info = {'eps': epsilon, 
                      'last_score': scores.pop(),
                      'solved_in': i_episode,
                      'last_100_avg': np.mean(scores_window),
                      'save_file': save_name}
    
    return scores, scores_mean, execution_info 