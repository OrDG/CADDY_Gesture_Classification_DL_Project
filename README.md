# CADDY_Gesture_Classification_DL_Project
This repository contains a project in Deep Learn created by MatanTP and I.
To run this project, you will need to download the full CADDY dataset (Complete dataset – 2.5GB, zipped) from here:
http://www.caddian.eu//CADDY-Underwater-Gestures-Dataset.html

Then you will need to upload the zipped dataset and the yaml file from this repo to your Google Drive under a folder named : 'CADDY_stereo_gesture_data'

After that, click here for our full project in google colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1n5I8w9td3rDZV2CCbk65F9akTgs7LpxD?usp=sharing)

We seperated this project into two parts:
1) A CNN which locates the hand of the diver - using YOLOv5 object detection architecture (for more info visit https://github.com/ultralytics/yolov5).
3) A CNN which classifies the gesture of the hand from the cropped image - using ower own architecture.
