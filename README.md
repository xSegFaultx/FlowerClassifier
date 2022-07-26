# FlowerClassifier
[![Udacity AI Programming with Python Nanodegree](https://img.shields.io/badge/Udacity-AI%20Programming%20with%20Python%20ND-deepskyblue?style=flat&logo=udacity)](https://www.udacity.com/course/ai-programming-python-nanodegree--nd089)
[![Pytorch](https://img.shields.io/badge/%20-Pytorch-grey?style=flat&logo=pytorch)](https://pytorch.org/)

The purpose of this project is to train a neural network that can recognize different species of flowers using the [flower dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) from the University of Oxford.

## Project Description
This project has 2 parts:
* First, implement and test the classifier in Jupyter Notebook (**only important steps are listed below. Please take a look at the notebook file if you want to see all the steps with detailed explanations**)
  + Load all the training, testing and validation data
  + Apply transformations on all the data to make sure each image has the same size. Also perform data augmentation on training data (like rotation and flip) which can help the network generalize
  + Build the classifier. Here I used a pretrained VGG16 as the feature extractor and three fully connected layers to "map" these features to the label
  + Train the network
  + Save the trained model in a checkpoint file for later inference
  + Perform inference on the trained network

* Second, convert the trained and tested classifier to a command line application
  + For this part of the project, most of the codes in func.py, network.py, predict.py and train.py are copied from the notebook version of the project, because they are pretty much the same.
  + The only difference is that predict.py and train.py can take command line input from users. Therefore, users can customize their network structure, choose their own hyper-parameters, etc.


## Data Transformations
There are two reasons why we need to perform transformations on our data. First, images in the dataset have different shapes, however, the VGG16 network that is used later only takes 244x244 pixels images as input. Therefore, we need to reshape all the images so that we can feed them into the network later. Second, randomly perform transformations like rotation, flip can increase the variety of samples and hence help the network generalize. Also, we need to normalize the image as VGG16 requires. (**More details of normalization can be found in the notebook**)    

So here is what I did:
  + Resize all images in training, testing and validation set to 244x244 pixels while keeping aspect ratio of the original images
  + Randomly rotate, horizontal flip and vertical flip images in the training set 
  + Normalize all the images as VGG16 requires


## Structure of the Network
For this project, we will build our network using transfer learning. I used a pre-trained VGG16 network as my feature extractor and 3 fully connected layers as the classifier that maps features to labels.
* VGG16
  + Since the VGG16 is pre-trained, we don't want to change the parameters in VGG16 during backpropagation. Therefore, we need to freeze the VGG16 network
* Classifier
  + Look at the structure of VGG16, the last convolutional block has output 25088. Therefore, the input layer size should be the same.
  + We have 102 different classes in total. Therefore, the output layer size should be 102.
  + For the hidden layer, I only used 1 layer of dense network with size 512. I started with 2 layers with size 8192 and 512 and soon found out that the training took a lot of time. Therefore, I deleted the layer with size 8192 to speed up the training. As a result, the training was faster and the final accuracy was slightly better. (I think this is because of reducing overfitting)
  + For the dropout rate I chose to use 0.2 after testing various rate from 0.1 to 0.5.

  
## Result
I only trained the network for 2 epochs and the accuracy of the classifier has already gone over 75% which is good enough according to the rubric (70%).
Next, I tested my model on the testing dataset.
Testing set is a data set that the classifier has never seen before. 
Testing our trained classifier on testing data set can give us a good estimate of the performance of the classifier on real-world data. The testing accuracy of my classifier is about 75% which exceeds the criteria set by the rubric (70%).


## Save and Load the Checkpoint
We can save our trained network and load it later for inference so that we don't need to re-train the network from scratch every time we need to inference on the network.
* Saving
  + We need to save the state of the trained network (values of all the parameters), so we don't need to train it again later
  + We need to save the structure information of the network (input size, hidden size, output size), so we can re-build the network later
  + Other necessary information for inferencing. For example, the mapping between real labels and one-hot encoded labels.
  + Lastly, we want to save all these into a checkpoint file
* Loading
  + Load the checkpoint file
  + Re-build the classifier by using the structure information we saved before
  + Load the state of the model back to the classifier we just create


## Inference on the Network
Before feeding the image to the network, we need to do some preparations.
* First, since users can input any sizes of images and VGG16 only accepts one input shape, we need to process user input images so that we can feed it into our network. (Basically the same thing we did on testing and validation set)
  + Resize all images in training, testing and validation set to 244x244 pixels while keeping aspect ratio of the original images
  + Normalize all the images as VGG16 requires
* Second, we need to put the network into inference mode
  + Put the network to evaluation mode so that layers like dropout, batchnorm (we didn't use in this project), etc. will not affect the result of inference
  + Turn auto-grad off since we will not perform backpropagation during inference. This can reduce memory usage and reduce a lot of computations.

After all these preparations, we can finally do inference on the network. For this project, we will show top k probable classes as our result.
Fig 1.0 below shows some of the predictions made by the model.
<figure>
<img src="https://github.com/xSegFaultx/FlowerClassifier/raw/master/assets/fig1.0.png" alt="Model Prediction">
<figcaption align = "center"><b>Fig.1.0 - Model Prediction </b></figcaption>
</figure>


## Command Line Version
The command line version is pretty much the same as the notebook version. The only difference is that the command line version allows users to customize their network structure, choose their own hyperparameters, etc.  

To build and train the network:
* Run: `python train.py data_directory` "data_directory" is where the training, testing and validation datasets are located. Below are all the optional inputs.
  + `--save_dir` The directory you want to save the checkpoint file. The default directory is checkpoints (type: String)
  + `--arch` Choose one of the pre-trained network. You can choose from vgg11, vgg13, vgg16, vgg19, densenet121, densenet161, densenet169, and densenet201. The default is vgg16 (type: string)
  + `--learning_rate` The learning rate of the neural network. The default is 0.0012 (type: float)
  + `--hidden_units` The number of inputs of the hidden layer in the classifier. The default is 512 (type: int)
  + `--epochs` Number of epochs. The default is 3 (type: int)
  + `--gpu` Turn on gpu training. The default is OFF (type: boolean)

To perform inference on the trained network:
* Run: `python predict.py /path/to/image checkpoint` "/path/to/image" is the path to the image that you want the network to classify and "checkpoint" is the path to the checkpoint file (.pth file). Below are all the optional inputs.
  + `--top_k` Show the top k possible classes with their possibility. The default is 1 (type: string)
  + `--category_names` A json file that contains a dictionary that can convert class indexes to category names. The default is None (type: string)
  + `--gpu` Turn on gpu predicting. The default is OFF (type: boolean)

# WARNING
This is a project from Udacity's ["AI Programming with Python Nanodegree"](https://www.udacity.com/course/ai-programming-python-nanodegree--nd089). 
I hope my code can help you with your project (if you are working on the same project as this one) but please do not copy my code and please follow [Udacity Honor Code](https://www.udacity.com/legal/community-guidelines) when you are doing your project.