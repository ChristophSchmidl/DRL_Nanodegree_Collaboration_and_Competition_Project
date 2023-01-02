# Udacity Nanodegree - Deep Reinforcement Learning

<img src="img/tennis.gif" width="650">

## Project 3: Collaboration and Competition

The third project in this Nanodegree is about collaboration and competition using the Unity ML-Agents Tennis Environment.

In this environment, two agents control rackets to bounce a ball over a net. 


### Project Details

If an agent hits the ball over the net, it receives a **reward of +0.1**. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a **reward of -0.01**. Thus, the goal of each agent is to keep the ball in play.

The **observation/state space** consists of **8 dimensions** corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. **Two continuous actions** are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an **average score of +0.5** (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the **average (over 100 episodes) of those scores is at least +0.5**.

### Getting Started

This repository was implemented with Python version 3.9.13. The following steps should enable you to reproduce and test the implementation.

- Create a python virtual environment at the root: ``python -m venv venv``
- Activate the virtual environment: ``source venv/bin/activate`` (if you are using Linux/Ubuntu)
- Upgrade pip: ``pip install --upgrade pip`` (optional)
- Install dependencies from local folder: ``pip install ./python``

After these instructions, everything should be ready to go. However, if you encounter compatibility issues with your CUDA version and Pytorch, then you could try to solve these problems by installing a specific PyTorch version that fits your CUDA version. In my case, I could resolve it using the following commands:

- ``pip uninstall torch``
- ``pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html``

The repository already contains the unity environments for Linux under the following locations:

-  ``src/Tennis_Linux`` (visual environment)
- ``src/Tennis_Linux_NoVis`` (non-visual environment)

However, if you want to install the unity environments for a different operating systems then you can find the download instructions below.


#### Download the Unity Environment

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)  (**Note**: You can replace ``Tennis_Linux.zip`` at the end of the URL with ``Tennis_Linux_NoVis.zip`` to get the non-visual environment. I already included both in the repo.)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)


### Instructions

After cloning the repository and installing all necessary dependencies, you can train and evaluate the different agents through the command line. The file ``src/main.py`` is using the ``argparse`` library to parse the command line arguments. You can use the following command to see all available arguments:

``python -m src.main --help``

which outputs the following:

```
ActorCritic methods - Collaboration and Competition Project

optional arguments:
  -h, --help            show this help message and exit
  -gpu GPU              GPU: 0 or 1. Default is 0.
  -episodes EPISODES    Number of games/episodes to play. Default is 5000.
  -alpha ALPHA          Learning rate alpha for the actor network. Default is 0.0001.
  -beta BETA            Learning rate beta for the critic network. Default is 0.001.
  -gamma GAMMA          Discount factor for update equation. Default is 0.99.
  -tau TAU              Update network parameters. Default is 0.001.
  -algo ALGO            You can use the following algorithms: DDPGAgent. Default is DDPGAgent.
  -buffer_size BUFFER_SIZE
                        Maximum size of memory/replay buffer. Default is 1000000.
  -batch_size BATCH_SIZE
                        Batch size for training. Default is 128.
  -load_checkpoint      Load model checkpoint/weights. Default is False.
  -model_path MODEL_PATH
                        Path for model saving/loading. Default is data/
  -plot_path PLOT_PATH  Path for saving plots. Default is plots/
  -use_eval_mode        Evaluate the agent. Deterministic behavior. Default is False.
  -use_multiagent_env   Using the multi agent environment version. Default is False.
  -use_visual_env       Using the visual environment. Default is False.
  -save_plot            Save plot of eval or/and training phase. Default is False.
```

#### Training

If you want to start training the agents from scratch with default hyperparamters, you can use the following command:

- ``python -m src.main -algo <AGENT_NAME> -episodes <NUMBER_OF_EPISODES>``

Note: At the moment only agent "DDPGAgent" is available.

#### Evaluation

If you want to evaluate the trained agents in non-visual mode (fast), you can use the following command:

- ``python -m src.main -algo <AGENT_NAME> -episodes <NUMBER_OF_EPISODES> -use_eval_mode``

The above command simply loads the appropriate model weights, sets the noise to 0.0 to enforce a deterministic behavior (no exploration, pure exploitation) and runs the agent in non-visual mode.

If you want to see the trained agents in action in visual mode, you can use the following command:

- ``python -m src.main -algo <AGENT_NAME> -episodes <NUMBER_OF_EPISODES> -use_eval_mode -use_visual_env``


### Report

If you are interested in the results and a more detailed report, please have a look at [report](REPORT.md).