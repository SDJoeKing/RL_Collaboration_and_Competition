[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"


# Udacity DRL Project 3 -  Collaboration and Competition


This is a deep reinforcement learning project with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment, based on [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents). The environment was compiled on toolkit version 0.4, provided by Udacity.

![Trained Agent][image1]

### Environment introduction

1. This environment involves two collaborative rackets hitting one ball, the aim is to hit the ball over the net from one racket to the other racket while maintaining the ball not dropping.
1. State and action space: The state space is `24`, consisting `8` variables corresponding to the `position` and `velocity` of the `ball` and `racket` (i.e. 8X3 space). Each racket receives its own, local observation. 

1. The action space is `2`, all continuous within range of `(-1, 1)`. The two continuous actions are corresponding to movement toward (or away from) the net, and jumping.orresponding to torques applicable to two joints.

2. Reward is `+0.1` for each step that one racket bounces the ball over the net, however, the agent will receive `-0.01` penalty if the ball drops to the gound or outbound. Thus, the goal of each racket is to keep the ball in play. For each time step, both rackets (or agents) will receive their local reward, added towards the score. When episode ends, the final `undiscounted score` for that episode is the maximum value of the two racket's score. 

3. **`Success criteria`: to solve the environment, the agent must get an average score of at least +0.5 over 100 consecutive episodes.**

4. Bench implemented by Udacity: the bench agent training performance provided by Udacity indicates maximum averaged score over 100 episodes are approximately 0.92, training completed at circa 2500 eposides. Although the algorithm quickly collapsed and became unrecoverable.  

### Getting Started adapted from Udacity instructions  

1. Copy this github repo:   
    ```
    git clone https://github.com/SDJoeKing/RL_Collaboration_and_Competition.git
    ```
3.  Download the environment from one of the links below.  You need only select the environment that matches your operating system:

     ```
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)
    ```
2. Place the file in the downloaded repo, and unzip (or decompress) the files. 
3. Activate the Python 36 virtual environment;
    ```
    At Linux terminal:
    source ~/your_python_36_env/bin/activation
    pip install numpy matplotlib scipy ipykernel pandas
    ```
5. While with python 36 activated, change to a different folder and setup the dependencies (noted gym environment is not required for this project):
     ```
    cd new_directory
    git clone https://github.com/udacity/deep-reinforcement-learning.git
    cd python
    pip install .
    ```
5. Due to platform and cuda restrictions, the requirement from step 5 of torch==0.4.0 can no longer be satisfied in my machine. Instead I have losen the restriction to any torch versions that are suitable for current machine. The rest of the requirements remain unchanged and satisfied (including the unityagents version requirement). 
6. Navigate back to the downloaded **this** repo and check `Tennis.ipynb`

### Instructions
In `Tennis.ipynb`, the training process was demonstrated. The `agent.py` and `model.py` contains codes for RL agent and the backend neural net architecture, respectively.  A report.md file is also included explaining the project.

