# FlowerClassifier

The purpose of this project is to build a neural network that can recognize different species of flowers using pytorch.

__WARNING__:This is a project from Udacity's " [AI Programming with Python Nanodegree](https://www.udacity.com/course/ai-programming-python-nanodegree--nd089)". I hope my code can help you with your project (if you are working on the same project as this one) but please do not copy my code and please follow [Udacity Honor Code](https://www.udacity.com/legal/community-guidelines) when you are doing your project.

## Project Description
This project has 2 parts:
* First, implement and test the classifier in Jupyter Notebook (**only important steps are listed below. Please take a look at the notebook file if you want to see all the steps with detailed explanations**)
  + Load all the training, testing and validation data
  + Apply tranformations on all the data to make sure each image has the same size. Also apply extra transformations on training data (like rotation and flip) which can help the network generalize
  + Build the classifier. Here I used a pretrained VGG16 to extra all the features from an image and three dense layers to "map" these features to the label
  + Train the network
  + Save the trained model in a checkpoint file for later inference
  + Perform inference on the trained network

* Second, convert the trained and tested classifier to a command line application
  + For this part of the project, most of the code in func.py, network.py, predict.py and train.py are copied from the notebook version of the project. Because, they are pretty much the same
  + The only difference is that predict.py and train.py can take command line input from users. Therefore, users can customize their network structure, choose their own hyper parameters, etc.




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
|           ./predict.py           | Use a trained network to predict the class for an input image |
|            ./test.jpg            | A randomly chosen image that is used to test the classifier  |
|            ./train.py            |     Train the network and save the model as a checkpoint     |

