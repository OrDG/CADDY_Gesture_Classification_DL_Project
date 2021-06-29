# Underwater Gesture Classification Deep Learning Project
This repository contains a project in Deep Learn created by @MatanTopel and I.

We seperated this project into two parts:
1) A CNN which locates a hand of a diver - using YOLOv5s object detection architecture (for more information visit https://github.com/ultralytics/yolov5).
3) A CNN which classifies the gesture of the hand from the cropped image - using our own architecture.

After training, the full network got 97.85% accuracy on the test set.

- Outline
  * [Background](#background)
  * [Chosen Solution](#Chosen-Solution)
  * [Results](#Results)
  * [Files in the repository](#files-in-the-repository)
  * [Usage](#Usage)
  * [References](#references)

## Background
### The CADDY project
* CADDY is a project focused on developing a robot that communicates with a diver and preforms tasks.
* CADDIAN is the sign language the diver uses to communicate with the robot.
Project site can be found here:
http://www.caddy-fp7.eu/
![CADDY project](https://github.com/OrDG/CADDY_Gesture_Classification_DL_Project/blob/main/CADDY.png)

* One of the challenges of CADDY is Interpreting the hand gestures of the diver from a big unclear picture.
* CNN is ideal for localization and for classifying images – translating CADDIAN to English!
### Our goal
creating a high accuracy CNN classifier of a diver’s gestures from CADDIAN, using stereo images taken underwater in 8 different water conditions:

![image](https://user-images.githubusercontent.com/35059685/123801212-bda00c00-d8f2-11eb-8a38-0904e3f01daa.png)

In this notebook, we will explain and implement the algorithm described in the paper. This is what we are trying to achieve:
![UP gesture](https://github.com/OrDG/CADDY_Gesture_Classification_DL_Project/blob/9e952de64e97299119253718d96dffbed731e982/final_classification/in2test.png)

![Succesfull classification](https://github.com/OrDG/CADDY_Gesture_Classification_DL_Project/blob/9e952de64e97299119253718d96dffbed731e982/final_classification/in2teslabel.png)

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

Results on the testset after training (images already resized and normalizied):

![image](https://user-images.githubusercontent.com/35059685/123801537-0ce63c80-d8f3-11eb-8105-17672f0c6681.png)

## Results

![image](https://user-images.githubusercontent.com/35059685/123802543-191ec980-d8f4-11eb-8f1f-c1237b961f19.png)

## Files in the repository
|File name         | Purpsoe |
|-----------------------|------|
|(1) `ProjectDL7.ipynb`|main program for training and merging both networks|
|(2) `caddy_loc.yaml`| file containing the directories of the train/valid/test for the YOLOv5 and the num of classes (1 in our case)|
|(3) `best.pt`| weights of the trained YOLOv5s on our dataset|
|(4) `caddy_cnn_ckpt1.pth`|weights of our trained CNN on our dataset|
|(5)`final_classification`|folder that contains the results of our traind networks|

## Usage
To use this project, you will need to download files and uplode them to a designated folder in your google drive named: 'CADDY_stereo_gesture_data'.

If you want to use the CADDY dataset, uplode the full CADDY dataset (Complete dataset – 2.5GB, zipped) from here:http://www.caddian.eu//CADDY-Underwater-Gestures-Dataset.html
 
 ### Using the trained networks
 To use our trained end-to-end network, you first need to download files (2)-(4) from this repo, and uplode them to the designated folder.
 
 Then you can use our example of running our fully trained network in google colab here:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fvR3Y70IWimrEnJupQt2Bw9aRX8lGdr2?usp=sharing)

 ### Training the networks
 To train our end-to-end network, you first need to download file (2) from this repo, and uplode them to the designated folder.
 
 After that, click here to use our full project in google colab:
 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1agVUMEALmVCe9zGvnj-YYn6BJnqgz-Ij?usp=sharing)

# Reference
[1] Lu C, Xu L, Jia J. Combining sketch and tone for pencil drawing production[C]//Proceedings of the Symposium on Non-Photorealistic Animation and Rendering. Eurographics Association, 2012: 65-73.

[2] Matlab implementation by "candtcat1992" - https://github.com/candycat1992/PencilDrawing
