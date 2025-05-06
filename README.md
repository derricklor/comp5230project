# comp5230project

# Computer Vision Face Detection and Recognition Project by Derrick Lor

## Overview

This is a computer vision project written in python using YOLOv11 for face detection and custom
python modules named hog.py and faceAlign.py. We train a custom model using the YOLO framework and
two datasets to detect faces. Then it is piped to the alignment portion of the code, which as the 
name suggests aligns or normalizes the images in a preprocessing step. All the faces have their eye
positions be absolutely level and resized. Then it goes through the HoG feature extractor which then
can be 1D vectorized to be able to calculate the distance metric either with Euclidean, cosine, or
correlation modes. The images in the database directory under whitelist or blacklist, will also 
follow the alignment and HoG extractor. Then the image is compared against the list of images in the
database. If the image distance is below the threshold found in the database, then the image is 
labeled the known face in the database. Otherwise it is unknown.


## Table of Contents
* [Tree View]
* [Installation]
* [Usage]
* [Configuration]
* [License](#license) (Optional)

## Tree View
project
├── datasets
│   ├── Faces
│   	├──  test
│   	├──  train
│   	├──  valid
│   	└── data.yaml
│   └── Face-Detect
│   	├──  test
│   	├── train
│   	├── valid
│   	└── data.yaml
├── known_faces
│   ├── blacklist
│   	└── face
│  			└── Tim.jpg
│   └── whitelist
│   	└── face
│   		├── Ben.jpg
│    		├── Derrick.jpg
│    		└── Josh.jpg
├── faceAlign.py
├── FaceDetectModelDemos.ipynb
├── FaceDetectModelEval.ipynb
├── FaceDetectModelTest.ipynb
├── hog.py
├── multiple_datasets.yaml
├── randomSelectImages.py
├── readme.txt
├── shape_predictor_68_face_landmarks.dat
├── yolo11n_FaceDetectModel_MD.pt
└── yolo11n_FaceDetectModel_MD20.pt


## Installation
How to install and set up the project.

Python 3.10 is required. Can either start up conda environment and install dependencies or install
in global path.
The prerequisites were installed on 4/27/25 using the latest version up to this date.
This can be done using pip commands: pip install <insert_here>
cmake
dlib
ultralytics
scikit-learn
scikit-image
imutils
opencv-python
numpy
os
time

Required files below are to be downloaded from the repository or found online.
Required .pt files: yolo11n_FaceDetectModel_MD.pt, yolo11n_FaceDetectModel_MD20.pt
Required .yaml files: multiple_datasets.yaml
Required .dat files: shape_predictor_68_face_landmarks.dat
Required custom .py files: hog.py, faceAlign.py, randomSelectImages.py
Required custom .ipynb files: FaceDetectModelDemos.ipynb, FaceDetectModelTest.ipynb, 
				FaceDetectModelEval.ipynb

Datasets used can be found using these links: 
1. https://universe.roboflow.com/huanhoahoe/facedetect-jb2ph
2. https://universe.roboflow.com/school-7y83u/faces-ylez0-9vot8
3. https://www.kaggle.com/datasets/vasukipatel/face-recognition-dataset/data
Datasets 1 and 2 were both used to train the yolo11n_FaceDetectModel_MD.pt, 
yolo11n_FaceDetectModel_MD20.pt models in FaceDetectModelDemos.ipynb, with the configuration data stored in the multiple_datasets.yaml. Dataset 3 was used only to evaluate the performance in FaceDetectModelEval.ipynb.

* This project was compiled using Windows 11 Operating System. With the help of Visual Studio Code
software development environment. 


## Usage
How to use the project.

The FaceDetectModelDemos.ipynb file: contains the necessary code needed to run demos of the face
detection and recognition system live. Also contains instructions for how to train the models.
There is no user interface other than the opencv2 window used in the demo.
During the demo, keys q, r, f, k, z, x, and c are mapped for different uses.
q = quit demo
r = reload known faces list
f = save cropped image of detected face in whitelist folder
k = save cropped image of detected face in blacklist folder
z = decrement the distance threshold
x = increment the distance threshold
c = change between cosine, correlation, and Euclidean distance metrics.

FaceDetectModelTest.ipynb: Used to test code for debugging and exploratory analysis.

FaceDetectModelEval.ipynb: Used to evaluate the trained yolo11n_FaceDetectModel_MD20.pt model for face
recognition. Uses randomSelectImages.py to randomly select images from dataset 3 and test with simple
evaluation loop. Counts the number of correct predictions against the truth label values.


## Configuration
How to modify multiple_datasets.yaml dataset settings.

Allow for the YOLO library to have access to the two datasets used for the training by passing the 
file path of multiple_datasets.yaml. If necessary, modify the multiple_datasets.yaml where the 
training datasets are found in your system. The datasets are not included with the repo and must be
downloaded separately.


## License 
MIT License

Copyright (c) 2025 Derrick Lor

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
* Specify the license under which the project is distributed (e.g., MIT, Apache 2.0, GPL).
* Include a link to the full license file if applicable.
