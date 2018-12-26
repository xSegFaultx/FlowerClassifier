from torchvision import models
from network import Classifier
from PIL import Image
import numpy as np
import torch
def build_model(arch,output_size,hidden_size,class_to_idx,drop_p=0.2):
    #load the pre-trained model
    build_model = 'models.'+arch+'(pretrained=True)'
    model = eval(build_model)
    #turn off autograd on pre-trained model
    for param in model.parameters():
       param.requires_grad = False
    #get the input size
    input_size = 0
    if arch[0]=='v':
        input_size = model.classifier[0].in_features
    else:
        input_size = model.classifier.in_features
    #build the classifier
    classifier = Classifier(input_size,output_size,hidden_size,drop_p)
    #switch to the new classifier
    model.classifier = classifier
    model.class_to_idx = class_to_idx
    return model,input_size

#build validation function
def validation(model, validloader, criterion, device):
    validation_loss=0
    accuracy=0
    for v_images, v_labels in validloader:
        #load images, labels into device
        v_images,v_labels=v_images.to(device), v_labels.to(device)
        v_output=model.forward(v_images)
        validation_loss+=criterion(v_output, v_labels).item()
        ps = torch.exp(v_output)
        equality = v_labels.data==ps.max(dim=1)[1]
        accuracy+=equality.type(torch.FloatTensor).mean()
    return validation_loss/len(validloader),accuracy/len(validloader)

#build model from checkpoint file
def load_model(path,device):
    #load the checkpoint file from disk
    if device ==torch.device('cpu'):
        checkpoint = torch.load(path,map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(path,map_location=lambda storage, loc: storage.cuda(0))
    #re-build the classifier
    classifier = Classifier(checkpoint['input_size'],checkpoint['output_size'],[checkpoint['hidden_size']],checkpoint['drop_p'])
    #load the pre-trained model
    arch = checkpoint['arch']
    build_model = 'models.'+arch+'(pretrained=True)'
    model = eval(build_model)
    for param in model.parameters():
        param.requires_grad=False
    #switch the classifier
    model.classifier = classifier
    #load the weights and bias back on
    model.load_state_dict(checkpoint['model_state'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model
#process the image
def process_image(image):
    #resize the image
    #find the min edge and max edge
    size = image.size
    ratio = 256/min(size)
    #resize the image
    image.thumbnail((max(size)*ratio,max(size)*ratio))
    #center crop the image
    width,height = image.size
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    image = image.crop((left,top,right,bottom))
    #convert to np array
    np_image = np.array(image)
    #type cast
    np_image = np_image.astype('float64')
    #scale color channel
    np_image/=255
    #normalize
    np_image -= np.array([0.485, 0.456, 0.406])
    np_image /= np.array([0.229, 0.224, 0.225])
    #Transpose
    np_image = np_image.transpose((2,0,1))
    return np_image

#use model to predict the image
def predict(image_path,model,topk,device):
    #prepare the image
    #open the image
    pil_image = Image.open(image_path)
    np_image = process_image(pil_image)
    #convert np array to tensor
    x=torch.from_numpy(np_image)
    #add batch size
    x.unsqueeze_(0)
    #load model from checkpoint
    model_p = load_model(model,device)
    #move model and data to gpu/cpu
    model_p.to(device)
    #convert x to float data type and move to gpu/cpu
    x = x.float().to(device)
    #pass through nn in evaluation mode with out auto grad
    model_p.eval()
    with torch.no_grad():
        output = model_p.forward(x)
    ps = torch.exp(output)
    #get the prob of the topk
    topk_p = ps.topk(topk)[0]
    #get the index of the topk
    topk_i = ps.topk(topk)[1]
    #invert the class_to_idx dict
    idx_to_class = {v:k for k,v in model_p.class_to_idx.items()}
    topk_c = [idx_to_class[i.item()] for i in topk_i[0]]
    return topk_p[0].tolist(), topk_c    