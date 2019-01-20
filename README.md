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
├── checkpoints
│   └── checkpoint.pth
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
|   ./checkpoints/checkpoint.pth   | A checkpoint file that saves the trained network (command line) |
|         ./checkpoint.pth         | A checkpoint file that saves the trained network (notebook)  |
|          ./flowers/test          |        This directory contains all the testing images        |
|         ./flowers/train          |       This directory contains all the training images        |
|         ./flowers/valid          |      This directory contains all the validation images       |
|            ./func.py             |   All the helper functions for ./predict.py and ./train.py   |
| ./Image Classifier Project.ipynb | All the code and details of the project in a jupyter notebook |
|           ./network.py           |        A class that defines the classifier (network)         |
|           ./predict.py           | Use a trained network to predict the class for an input image |
|            ./test.jpg            | A randomly chosen image that is used to test the classifier  |
|            ./train.py            |     Train the network and save the model as a checkpoint     |

## Data Transformations
There are two reasons why we need to perform transfermations on our data. First, images in the dataset have different shapes, however, the VGG16 network that is used later only takes 244x244 pixels images as input. Therefore, we need to reshape all the images so that we can feed them into the network later. Second, randomly perform tranformations like rotation, flip can increase the variation of samples and hence help the network generalize. Also, we need to normalize the image as VGG16 requires. (**More details of normalization can be found in the notebook**)  

So here is what I did:
  + Resize all images in training, testing and validation set to 244x244 pixels while keeping aspect ratio of the original images
  + Randomly rotate, horizontal flip and vertial flip images in the training set (I didn't do the same to testing and validation set because these two sets will not affect the performance of the network and these two sets should represent real world data as closely as possible)
  + Normalize all the images
  
## Structure of the network
## Traning and Hyper Parameters
I only trained the network for 2 epochs and the accuracy of the classifier 
## Testing result
## Inference on the network
## Command line version
