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



## Underwater Gesture Classification Deep Learning Project
Convolutional neural network designed to classify images of a diver hand signing the intended word.
![alt text](https://github.com/OrDG/CADDY_Gesture_Classification_DL_Project/blob/9e952de64e97299119253718d96dffbed731e982/final_classification/in2test.png)

![alt text](https://github.com/OrDG/CADDY_Gesture_Classification_DL_Project/blob/9e952de64e97299119253718d96dffbed731e982/final_classification/in2teslabel.png)
### Inspired on the the CADDY project developing a diving assiting robot - the diver uses the CADDIAN sign language to commmunicate with the robot
Project site can be found here:
http://www.caddy-fp7.eu/
![alt text](https://github.com/OrDG/CADDY_Gesture_Classification_DL_Project/blob/main/CADDY.png)

![alt text](https://github.com/OrDG/CADDY_Gesture_Classification_DL_Project/blob/main/tut_track_anim.gif)
We worked on an open access dataset provided by them at the website: 
Draws inspiration from the Matlab implementation by "candtcat1992" - https://github.com/candycat1992/PencilDrawing

In this notebook, we will explain and implement the algorithm described in the paper. This is what we are trying to achieve:
![alt text](https://github.com/taldatech/image2pencil-drawing/blob/master/images/ExampleResult.JPG)

We can divide the workflow into 2 main steps:
1. Pencil stroke generation (captures the general strucure of the scene)
2. Pencil tone drawing (captures shapes shadows and shading)

Combining the results from these steps should yield the desired result. The workflow can be depicted as follows:
![alt text](https://github.com/taldatech/image2pencil-drawing/blob/master/images/Workflow.JPG)

* Both figures were taken from the original paper

Another example:
![alt text](https://github.com/taldatech/image2pencil-drawing/blob/master/images/jl_compare.JPG)

# Usage
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
# Parameters
* kernel_size = size of the line segement kernel (usually 1/30 of the height/width of the original image)
* stroke_width = thickness of the strokes in the Stroke Map (0, 1, 2)
* num_of_directions = stroke directions in the Stroke Map (used for the kernels)
* smooth_kernel = how the image is smoothed (Gaussian Kernel - "gauss", Median Filter - "median")
* gradient_method = how the gradients for the Stroke Map are calculated (0 - forward gradient, 1 - Sobel)
* rgb = True if the original image has 3 channels, False if grayscale
* w_group = 3 possible weight groups (0, 1, 2) for the histogram distribution, according to the paper (brighter to darker)
* pencil_texture_path = path to the Pencil Texture Map to use (4 options in "./pencils", you can add your own)
* stroke_darkness = 1 is the same, up is darker.
* tone_darkness = as above

# Folders
* inputs: test images from the publishers' website: http://www.cse.cuhk.edu.hk/leojia/projects/pencilsketch/pencil_drawing.htm
* pencils: pencil textures for generating the Pencil Texture Map

# Reference
[1] Lu C, Xu L, Jia J. Combining sketch and tone for pencil drawing production[C]//Proceedings of the Symposium on Non-Photorealistic Animation and Rendering. Eurographics Association, 2012: 65-73.

[2] Matlab implementation by "candtcat1992" - https://github.com/candycat1992/PencilDrawing
