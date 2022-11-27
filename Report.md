## Report

### Learning Algorithm

For this project, Multiagent DDPG was experimented first using the method of ##centralised training decentralised execution##, which means that the critic nets have all the observations and actions information for evaluating the action performance, and policy nets have access to only local observations. However, the results were not promising which perhaps owing to the inefficient implementation. I then followed the hint given from the original Udacity site where one central agent was trained, to control the two rackets with shared memory. This method could work partly due to the relatively simple environment, small number of controlling objects (i.e. 2) and the nature of collaboration in contrast to competition.

Therefore, in this project, I still adopted previously very successful TD3 ([Twin Delayed DDPG](https://spinningup.openai.com/en/latest/algorithms/td3.html)) algorithm, but with some strucutral changes to enhance the efficiency. Standard TD3 tricks including `Clipped double Q learning`, `Delayed policy update` and `Target policy smoothing` are implemented to stablise the training in comparison to standard DDPG algorithm. Meanwhile, the structure of Critic nets are changed such that one Critic outputs two `Q` values, which is quite neat with TD3 which contains two sets of target-local Critic nets, thanks to the auther of `TD3`'s repo [here](https://github.com/sfujim/TD3/blob/master/TD3.py).

Clipped double Q learning resembles the double-q learning concepts which utilises two Q networks, and another two targeted Q nets for each. When evaluating the targeted Q value, the smaller Q values from the targeted Q nets are used to prevent Q nets exploiting ocassional large Q values. Delayed policy udpate entails updating policy nets only after several Q nets update to stablise the policy learning. And finally the targeted policy smoothing refers to adding uncorrelated Gaussian noises on targeted actions when computing the target actions from target policy. Noise is also added to actions generated from policy during training, to encourage exploration.
 
### Hyper parameters selection
For the agent, I have used the following parameters:
  ```
    states = 24, 
    actions = 2, 
    gamma = 0.99, 
    lr = 5e-4, 
    tau = 0.005, 
    policy_smooth_noise = 0.2, 
    noise_clip = 0.5, 
    policy_delay = 2, 
    batch_size = 256, 
    seed = 33
    start_steps = 10000
 
  ```
The `start_steps` parameter is set for the environment to take random actions for a while to provide experiences into the replay buffer. It has been observed that this usually takes approximately `400` episodes. Therefore the final number of episodes to complete training will need to subtract this value.  

For training, a action noise of `0.3` is added for exploration and `0.2` clipped by `0.5` for target policy smoothing. 

There are no maximum iteration per episode limitation. However, at later training stages the agent becomes too good at the game that it drastically slows down the training, therefore, only `4500` episodes of training is allowed. 

For training function:
  ```
    episodes = 4500, 
    exploration = 0.3,  
    print_every = 200, 
    term_reward = 0.5, 
    seed = 10
  ```
### Neural Network architecture
For the underlying Actor NN models, it is defaulted to have two hidden layers, both `128` dimensions. The activation function is `ReLU` in between layers, and a `Tanh` function at the output to limit the policy to action space of (-1, 1).

For Critic NN model, as described above, two outputs corresponding to `Q1` and `Q2` are generated following two sets of identical strucutre, which are `128-128` layers. 

### Results discussion
The training with a single Agent solved the environment as reported, in episode `3933`, giving undiscounted score of `0.5043` averaged for over `100` episodes. Considering the first `400` episodes are only pure exploration with random actions, the actual episodes used for training is approximately `3500`. Additionally, the agent seems very stable, which is one of the advantages of TD3 agents. No sign of collapsing can be observed by `4500` episodes.

**From the figure of result below, it is evident that the episodes training required to achieve average score of +0.5 are 3933 episodes. The actual episodes used for training was approximately 3500**

 ![Training Results]()


### Future improvements
As discussed above, firstly I can try implementing `MADDPG` or extend it to `MA-TD3` agents. Additionally, prioritised experience replay should be effective, combining with the `10000` start_steps. Other ideas are to experiment with `PPO` based methods. 
