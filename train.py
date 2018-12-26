import sys
import os
import argparse
import torch
import func
from torch import nn
from torch import optim
from torchvision import datasets, transforms
#deal with all the input parameters
data_dir = sys.argv[1]
parser = argparse.ArgumentParser()
parser.add_argument('--save_dir',action = 'store',dest = 'save_dir',type = str, default = 'checkpoints', help = "The directory you want to save the checkpoint file, default directory is checkpoints")
parser.add_argument('--arch',action = 'store',dest = 'arch',type = str, default = 'vgg16',help = "Choose one of the pre-trained network. You can choose from vgg11,vgg13,vgg16,vgg19,densenet121,densenet161,densenet169,densenet201,default is vgg16")
parser.add_argument('--learning_rate',action = 'store',dest = 'lr',type = float, default = 0.0012,help = "The learning rate of the neural network, default is 0.0012")
parser.add_argument('--hidden_units',action = 'store',dest = 'hidden_size',type = int, default = 512, help = "The number of inputs of the hidden layer in the classifier, default is 512")
parser.add_argument('--epochs',action = 'store',dest = 'epochs',type = int, default = 3, help = "Number of epochs, default is 3")
parser.add_argument('--gpu',action = 'store_true',dest = 'gpu' , default = False, help = "Turn on gpu training, default is OFF")
params = parser.parse_args(sys.argv[2:])
save_dir,arch,learn_rate,hidden_size,epochs,gpu = params.save_dir,params.arch,params.lr,params.hidden_size,params.epochs,params.gpu
#load the training and validation data
#get dir name for train and validation
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
#Training set: rotate, resize,crop,horizontal, vertical flip, and normalize
train_transforms = transforms.Compose([transforms.RandomRotation(30),transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),
                                      transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
#validation set: resize,crop and normalize
validation_transforms = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),
                                          transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_set=datasets.ImageFolder(train_dir,transform=train_transforms)
validation_set=datasets.ImageFolder(valid_dir,transform=validation_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader=torch.utils.data.DataLoader(train_set,batch_size=32,shuffle=True)
validloader=torch.utils.data.DataLoader(validation_set,batch_size=32)

#get the mapping between class and index
class_to_idx = train_set.class_to_idx
#get the number of classes to determine the output size
output_size = len(class_to_idx.keys())
#create the model
model,input_size = func.build_model(arch,output_size,[hidden_size], class_to_idx)
#define optimizer and criterion
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(),lr=learn_rate)
#prepare for tarining 
print_every = 30
step = 0
#determine which device to run on
device = torch.device('cuda:0' if gpu else 'cpu')
#move model to the device
model = model.to(device)
#traning network
for e in range(epochs):
    running_loss=0
    for images, labels in trainloader:
        step+=1
        #move images, labels to device
        images, labels=images.to(device),labels.to(device)
        #clean grad
        optimizer.zero_grad()
        output=model.forward(images)
        loss=criterion(output, labels)
        loss.backward()
        optimizer.step()
        #calculate the loss
        running_loss+=loss.item()
        #print out the process
        if step%print_every==0:
            #evaluation mode
            model.eval()
            #no grad when doing validation
            with torch.no_grad():
                valid_loss, accuracy = func.validation(model,validloader, criterion, device)
            print('Epochs:{}/{}  '.format(e+1,epochs), "Running Loss:{:.4f}  ".format(running_loss/print_every),
                 "Validation Loss:{:.4f}  ".format(valid_loss),"Accuracy:{:.4f}".format(accuracy))
            running_loss=0
            model.train()
#save the model to disk
#create the checkpoint dict
checkpoint = {'arch':arch,'model_state':model.state_dict(),'input_size':input_size,'output_size':output_size,
             'hidden_size': hidden_size, 'class_to_idx':model.class_to_idx, 'drop_p':0.2}
#save to disk
#check if the dir exist, if not create it
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
path = save_dir+'/checkpoint.pth'
torch.save(checkpoint, path)
print("Traning Completed. Checkpoint file is saved to "+path)