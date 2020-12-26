# Solving Navigation using Deep Reinforcement Learning (Value-Based Methods)

## Introduction

The goal in this project is to create and train an agent to navigate and collect only yellow banana in a large, square world. By applying deep reinforcement learning to this environment, the agent essentially learns to collect only yellow bananas while avoiding blue ones. (I wouldnt get near a blue banana either :O) Read my blog on (Medium)[https://ahtchow.medium.com/solving-navigation-using-deep-reinforcement-learning-value-based-methods-3fe74fe85876] for this project!


![](./imgs/collecting_bananas.gif)

The application of this project is targeted for, while not limited to, robotic navigation. Some ideas that one could build off of this project are:
- Modern Day:
    1. Garbage Collector Robot
    2. Argricultural Robots (Muti-purpose)
    
- Futuristic:
    1. Autonomous vehicles 
    2. Humanoids



## Understanding the environment

**Summary**

In this environment, an Agent navigates a large, square world collecting bananas. Each episode of this task is limited to 300 steps. A reward of **+1** is provided for collecting a yellow banana, and a reward of **-1** is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible, avoiding the blue ones.

**RL Problem Specifications**


    
-    **Goal of Agent** : Collect yellow bananas, avoid blue bananas 
-    **Rewards** : ```+1``` collecting yellow bananas, ```-1```  collecting blue bananas
-    **Action Space** - Discrete, 4 Actions
-    **State Space** - Continious, 37-Dimension
-    **Solving Condition** : Average score of +13 over 100 consecutive episodes. 

**More on State and action spaces**

The state-space has **37 dimensions** and contains the agent's velocity, along with the ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. **Four discrete actions** are available, corresponding to:

- `0` - move forward
- `1` - move backward
- `2` - turn left
- `3` - turn right

### Concepts and Resources used for this Project

Notes:

 * [Deep Q-Networks](notes/Deep_Q-Networks.pdf)
 * [Deep Q-Networks Improvements](notes/Deep_Q-Learning_Improvements.pdf)


Academic Resouce Papers:

* [Deep Q-Network](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
* [Double Deep Q-Network](https://arxiv.org/abs/1509.06461)
* [Dueling Q-Network](https://arxiv.org/abs/1511.06581)
* [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)


## Files Included in this repository

* The code used to create and train the Agent
  * Navigation.ipynb : Driver code to run agent in environment!
  * agent.py : deep reinforcment agent
  * model.py : agent's models used for DRL
  * prioritized_replay_buffer.py
  * utils/replay_buffer.py : Replay buffer data structure
  * utils/prioritized_replay_buffer.py : Prioritized replay buffer data structure
* A file describing all the packages required to set up the environment
  * requirements.txt
* The trained model
  * baseline_params.pth : Params for Vanilla DQN
  * DDQN_params.pth : Params for Double DQN
  * PER_params.pth : Params for Prioritized Experience Replay DQN
  * dueling_params.pth : Params for Dueling DQN

* This README.md file

## Setting up the environment

This section describes how to get the code for this project and configure the environment.

### Getting the code
You have two options to get the code contained in this repository:
##### Option 1. Download it as a zip file

* [Click here](https://github.com/ahtchow/Reinforcement-Learning-Navigation/archive/master.zip) to download all the content of this repository as a zip file
* Uncompress the downloaded file into a folder of your choice

##### Option 2. Clone this repository using Git version control system
If you are not sure about having Git installed in your system, run the following command to verify that:

```
$ git --version
```
If you need to install it, follow [this link](https://git-scm.com/downloads) to do so.

Having Git installed in your system, you can clone this repository by running the following command:

```
$ git clone https://github.com/ahtchow/Reinforcement-Learning-Navigation.git
```

### Configuring the environment
The `requirements.txt` file included in this repository describes all the packages and dependencies required to set up the environment. 

It is recommended that you [create a new conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html), then use it to install ```pip install -r requirements.txt``` 

## How to train the Agent
The environment you have just set up has the files and tools to allow the training of the agent.  

Start the Jupyter Notebook server by running the commands below. A new browser tab will open with a list of the files in the current folder.

You must to set your operational system by downloading the appropriate executable coresponding to your OS. After you do this, drag and drop the correct files into the directory. (See Notebook for more instructions and clarificaiton)

The options available are:

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)

* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)

* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)

* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)


Use the jupyer notebook, Navigation.ipynb to run the models against the Unity Environment.


### Additional Notes

This environment has been built using the **Unity Machine Learning Agents Toolkit (ML-Agents)**, which is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents. You can read more about ML-Agents by perusing the [GitHub repository](https://github.com/Unity-Technologies/ml-agents).  

The project environment is provided by Udacity and is similar to, but not identical to the Banana Collector environment on the [Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector).  
