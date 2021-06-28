# CADDY_Gesture_Classification_DL_Project
This repository contains a project in Deep Learn created by @MatanTopel and I.
To run this project, you will need to download the full CADDY dataset (Complete dataset – 2.5GB, zipped) from here:
http://www.caddian.eu//CADDY-Underwater-Gestures-Dataset.html

Then you will need to upload the zipped dataset and the yaml file from this repo to your Google Drive under a folder named : 'CADDY_stereo_gesture_data'

After that, click here for our full project in google colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1n5I8w9td3rDZV2CCbk65F9akTgs7LpxD?usp=sharing)

We seperated this project into two parts:
1) A CNN which locates the hand of the diver - using YOLOv5s object detection architecture (for more information visit https://github.com/ultralytics/yolov5).
3) A CNN which classifies the gesture of the hand from the cropped image - using our own architecture.

After training, the full network got 97.47% accuracy on the test set.

![alt text](Network_Architecture.jpg?raw=true)


# Underwater Gesture Classification Deep Learning Project
Convolutional neural network designed to classify images of a diver hand signing the intended word.
### Inspired on the the CADDY project developing a diving assiting robot - the diver uses the CADDIAN sign language to commmunicate with the robot
Project site can be found here:
http://www.caddy-fp7.eu/
![CADDY project](https://github.com/OrDG/CADDY_Gesture_Classification_DL_Project/blob/main/CADDY.png)

We worked on an open access dataset provided by them at the website: http://www.caddian.eu/

In this notebook, we will explain and implement the algorithm described in the paper. This is what we are trying to achieve:
![UP gesture](https://github.com/OrDG/CADDY_Gesture_Classification_DL_Project/blob/9e952de64e97299119253718d96dffbed731e982/final_classification/in2test.png)

![Succesfull classification](https://github.com/OrDG/CADDY_Gesture_Classification_DL_Project/blob/9e952de64e97299119253718d96dffbed731e982/final_classification/in2teslabel.png)

- [Project](#Underwater Gesture Classification Deep Learning Project)
  * [Background](#background)
  * [Prerequisites](#prerequisites)
  * [Files in the repository](#files-in-the-repository)
  * [API (`ls_dqn_main.py --help`)](#api---ls-dqn-mainpy---help--)
  * [Playing](#playing)
  * [Training](#training)
  * [Playing Atari on Windows](#playing-atari-on-windows)
  * [TensorBoard](#tensorboard)
  * [References](#references)
## Background
The idea of this algorithm is to combine between Deep RL (DRL) to Shallow RL (SRL), where in this case, we use Deep Q-Learning (DQN) as the DRL algorithm and
Fitted Q-Iteration (FQI) as the SRL algorithm (which can be approximated using least-squares, full derivation is in the original paper).
Every N_DRL (number of DQN backprop steps) we apply LS-UPDATE to the very last layer of the DQN, by using the complete Replay Buffer, a fetaure augmentation technique and
Bayesian regularization (prevents overfitting, makes the LS matrix invertible) to solve the FQI equations.
The assumptions are that the features extracted from the last layer form a rich representation, and that the large batch size used by the SRL algorithm improves stability and performance.
In this work we added the derivations and approximations for Dueling DQN architecture and also added Boosted FQI as an optional SRL algorithm.
For full derivations and theory, please refer to the original paper.
## Prerequisites
|Library         | Version |
|----------------------|----|
|`Python`|  `3.5.5 (Anaconda)`|
|`torch`|  `0.4.1`|
|`gym`|  `0.10.9`|
|`tensorboard`|  `1.12.0`|
|`tensorboardX`|  `1.5`|
## Files in the repository
|File name         | Purpsoe |
|----------------------|------|
|`ls_dqn_main.py`| general purpose main application for training/playing a LS-DQN agent|
|`pong_ls_dqn.py`| main application tailored for Atari's Pong|
|`boxing_ls_dqn.py`| main application tailored for Atari's Boxing|
|`dqn_play.py`| sample code for playing a game, also in `ls_dqn_main.py`|
|`actions.py`| classes for actions selection (argmax, epsilon greedy)|
|`agent.py`| agent class, holds the network, action selector and current state|
|`dqn_model.py`| DQN classes, neural networks structures|
|`experience.py`| Replay Buffer classes|
|`hyperparameters.py`| hyperparameters for several Atari games, used as a baseline|
|`srl_algorithms.py`| Shallow RL algorithms, LS-UPDATE|
|`utils.py`| utility functions|
|`wrappers.py`| DeepMind's wrappers for the Atari environments|
|`*.pth`| Checkpoint files for the Agents (playing/continual learning)|
|`Deep_RL_Shallow_Updates_for_Deep_Reinforcement_Learning.pdf`| Writeup - theory and results|
## API (`ls_dqn_main.py --help`)
You should use the `ls_dqn_main.py` file with the following arguments:
|Argument                 | Description                                 |
|-------------------------|---------------------------------------------|
|-h, --help       | shows arguments description             |
|-t, --train     | train or continue training an agent  |
|-p, --play    | play the environment using an a pretrained agent |
|-n, --name       | model name, for saving and loading |
|-k, --lsdqn	| use LS-DQN (apply LS-UPDATE every N_DRL), default: false |
|-j, --boosting| use Boosted-FQI as SRL algorithm, default: false |
|-u, --double| use double dqn, default: false|
|-f, --dueling| use dueling dqn, default: false |
|-y, --path| path to agent checkpoint, for playing |
|-m, --cond_update| conditional ls-update: update only if ls weights are better, default: false |
|-e, --env| environment to play: pong, boxing, breakout, breakout-small, invaders |
|-d, --decay_rate| number of episodes for epsilon decaying, default: 100000 |
|-o, --optimizer| optimizing algorithm ('RMSprop', 'Adam'), deafult: 'Adam' |
|-r, --learn_rate| learning rate for the optimizer, default: 0.0001 |
|-g, --gamma| gamma parameter for the Q-Learning, default: 0.99 |
|-l, --lam| regularization parameter value, default: 1, 10000 (boosting) |
|-s, --buffer_size| Replay Buffer size, default: 1000000 |
|-b, --batch_size| number of samples in each batch, default: 128 |
|-i, --steps_to_start_learn| number of steps before the agents starts learning, default: 10000 |
|-c, --target_update_freq| number of steps between copying the weights to the target DQN, default: 10000 |
|-x, --record| Directory to store video recording when playing (only Linux) |
|--no-visualize| if typed, render the environment when playing, default: True (does not visualize) |
|--no-visualize| if not typed, render the environment when playing |

## Playing
Agents checkpoints (files ending with `.pth`) are saved and loaded from the `agent_ckpt` directory.
Playing a pretrained agent for one episode:

`python ls_dqn_main.py --play -e pong -y ./agent_ckpt/pong_agent.pth --no-visualize`
`python ls_dqn_main.py --play -e pong -y ./agent_ckpt/pong_agent.pth`

If the checkpoint was trained using Dueling DQN:

`python ls_dqn_main.py --play -e pong -f -y ./agent_ckpt/pong_agent.pth --no-visualize`
`python ls_dqn_main.py --play -e pong -f -y ./agent_ckpt/pong_agent.pth`

## Training

Examples:
* `python ls_dqn_main.py --train --lsdqn -e boxing -l 10 -b 64`
* `python ls_dqn_main.py --train --lsdqn --boosting --dueling -m -e boxing -l 1000 -b 64`
For full description of the flags, see the full API.
## Playing Atari on Windows
You can train and play on a Windows machine, thanks to Nikita Kniazev, as follows from this post on [stackoverflow](https://stackoverflow.com/questions/42605769/openai-gym-atari-on-windows/46739299):
`pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py` 
## TensorBoard
TensorBoard logs are written dynamically during the runs, and it possible to observe the training progress using the graphs. In order to open TensoBoard, navigate to the source directory of the project and in the terminal/cmd:
`tensorboard --logdir=./runs`
* make sure you have the correct environment activated (`conda activate env-name`) and that you have `tensorboard`, `tensorboardX` installed.
## References
* [PyTorch Agent Net: reinforcement learning toolkit for pytorch](https://github.com/Shmuma/ptan) by [Max Lapan](https://github.com/Shmuma)
* Nir Levine, Tom Zahavy, Daniel J. Mankowitz, Aviv Tamar, Shie Mannor [Shallow Updates for Deep Reinforcement Learning](https://arxiv.org/abs/1705.07461), NIPS 2017

* tone_darkness = as above

# Folders
* inputs: test images from the publishers' website: http://www.cse.cuhk.edu.hk/leojia/projects/pencilsketch/pencil_drawing.htm
* pencils: pencil textures for generating the Pencil Texture Map

# Reference
[1] Lu C, Xu L, Jia J. Combining sketch and tone for pencil drawing production[C]//Proceedings of the Symposium on Non-Photorealistic Animation and Rendering. Eurographics Association, 2012: 65-73.

[2] Matlab implementation by "candtcat1992" - https://github.com/candycat1992/PencilDrawing
