import sys
import argparse
import func
import torch
import json
from PIL import Image

image_path = sys.argv[1]
checkpoint_path = sys.argv[2]
parser = argparse.ArgumentParser()
parser.add_argument('--top_k',action = 'store',dest = 'top_k',type = int, default = 1, help = "Show the top k possible class with their possibility, default is 1")
parser.add_argument('--category_names',action = 'store',dest = 'category_names',type = str, default = None, help = "A json file that contains a dictionary that can convert class indexes to category names, default is None")
parser.add_argument('--gpu',action = 'store_true',dest = 'gpu' , default = False, help = "Turn on gpu predicting, default is OFF")
params = parser.parse_args(sys.argv[3:])
topk,category_names,gpu = params.top_k,params.category_names,params.gpu
#setup device
device = torch.device('cuda:0' if gpu else 'cpu')
#do the prediction
probs,classes = func.predict(image_path,checkpoint_path,topk,device)
#see if there is a class to name convert available
if category_names!=None:
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
        classes = [cat_to_name[cat] for cat in classes]
#print out the result
for c,p in zip(classes,probs):
    print('This image is classified as {} with a probablity of {:.4f}'.format(c,p))


    