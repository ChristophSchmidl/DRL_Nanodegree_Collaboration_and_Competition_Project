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

#### Download the Unity Environment

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)  (**Note**: You can replace ``Tennis_Linux.zip`` at the end of the URL with ``Tennis_Linux_NoVis.zip`` to get the non-visual environment. I already included both in the repo.)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)


### Instructions



#### Training



#### Evaluation




### Report

If you are interested in the results and a more detailed report, please have a look at [report](REPORT.md).