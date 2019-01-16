# FlowerClassifier

The purpose of this project is to build a neural network that can recognize different species of flowers using pytorch.

__WARNING__:This is a project from Udacity's " [AI Programming with Python Nanodegree](https://www.udacity.com/course/ai-programming-python-nanodegree--nd089)". I hope my code can help you with your project (if you are working on the same project as this one) but please do not copy my code and please follow [Udacity Honor Code](https://www.udacity.com/legal/community-guidelines) when you are doing your project.

## File structure of the project
Here this the file structure of this project (**only important files are listed**)
<pre>
FlowerClassifier
├── assets
├── cat_to_name.json
├── checkpoint.pth
├── flowers
│   ├── test
│   ├── train
│   └── valid
├── func.py
├── Image Classifier Project.ipynb
├── network.py
├── predict.py
├── README.md
├── test.jpg
└── train.py
</pre>
Description of some of the important files/directories

|            File name             |                         Description                          |
| :------------------------------: | :----------------------------------------------------------: |
|        ./cat_to_name.json        |        A mapping from category label to category name        |
|         ./checkpoint.pth         |       A checkpoint file that saves the trained network       |
|          ./flowers/test          |        This directory contains all the testing images        |
|         ./flowers/train          |       This directory contains all the training images        |
|         ./flowers/valid          |      This directory contains all the validation images       |
|            ./func.py             |   All the helper functions for ./predict.py and ./train.py   |
| ./Image Classifier Project.ipynb | All the code and details of the project in a jupyter notebook |
|           ./network.py           |        A class that defines the classifier (network)         |
|           ./predict.py           | The code that is used to doing inference on the trained network |
|            ./test.jpg            | A randomly chosen image that is used to test the classifier  |
|            ./train.py            |          The code that is used to train the network          |





