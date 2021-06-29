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

- Outline
  * [Background](#background)
  * [Chosen Solution](#Chosen-Solution)
  * [Prerequisites](#prerequisites)
  * [Files in the repository](#files-in-the-repository)
  * [Usage](#Usage)
  * [Playing](#playing)
  * [Training](#training)
  * [Playing Atari on Windows](#playing-atari-on-windows)
  * [TensorBoard](#tensorboard)
  * [References](#references)
## Background
* CADDY is a project focused on developing a robot that communicates with a diver and preforms tasks.
* CADDIAN is the sign language the diver uses to communicate with the robot.
* One of the challenges of CADDY is Interpreting the hand gestures of the diver from a big unclear picture.
* CNN is ideal for localization and for classifying images – translating CADDIAN to English!

Our goal is creating a high accuracy CNN classifier of a diver’s gestures from CADDIAN, using stereo images taken underwater in different water conditions.

## Chosen Solution
Localization - YOLOv5s
classification - Our own CNN.

### YOLOv5s
There is no published articles on YOLOv5, so we will show the architecture of YOLOv4, because it has many similarities to YOLOv5.

![image](https://user-images.githubusercontent.com/35059685/123794270-56328e00-d8eb-11eb-95f8-9b4c86da7dc7.png)

Results on the testset after training:

![image](https://github.com/OrDG/CADDY_Gesture_Classification_DL_Project/blob/8a7a5ea88b83ebe45d86709adb944f28d74d9b7b/final_classification/test_batch1_pred.jpg)

### Our own CNN
Our CNN architecture is conventional. The network is built from 3 connected Conv blocks which increase the number of channels and decrease the size of each channel, and at the end are connected to three Fully Connected layers, and a Softmax for classification after that.  We use RelU activations and Adam optimizer. 

![image](https://user-images.githubusercontent.com/35059685/123799809-4158f900-d8f1-11eb-9dc5-e85f437deda9.png)


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
 ## Usage
```python
from PencilDrawingBySketchAndTone import *
import matplotlib.pyplot as plt
ex_img = io.imread('./inputs/11--128.jpg')
pencil_tex = './pencils/pencil1.jpg'
ex_im_pen = gen_pencil_drawing(ex_img, kernel_size=8, stroke_width=0, num_of_directions=8, smooth_kernel="gauss",
                       gradient_method=0, rgb=True, w_group=2, pencil_texture_path=pencil_tex,
                       stroke_darkness= 2,tone_darkness=1.5)
plt.rcParams['figure.figsize'] = [16,10]
plt.imshow(ex_im_pen)
plt.axis("off")
```

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
