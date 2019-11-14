import json
import time

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from collections import OrderedDict

def load_data():
    """
        Purpose: Transform and load the images
        Inputs: None
        Outputs: Dictionary of image datasets
                 Dictionary of dataloaders
                 Mapping of category to name
    """
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Transform & Normalize Images
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])
    
    
    # Load the datasets
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    image_datasets = {'train':train_data,
                      'test':test_data,
                      'valid':validation_data}
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=32, shuffle=True)
    dataloaders = {'train':trainloader,
                   'test':testloader,
                   'validation':validationloader}
    
    with open('cat_to_name.json', 'r') as f:
        label_map = json.load(f)
        
    return image_datasets, dataloaders, label_map

def load_model(save_dir,gpu):
    """
    Purpose: Load a saved model
    Inputs:
        - Directory of where model is saved
        - String of architecture
        - Int of number of hidden units
        - Binary of GPU
    Outputs: None
    """
    if gpu == True:
        checkpoint = torch.load(save_dir)
    else:
        checkpoint = torch.load(save_dir,map_location="cpu")

    if checkpoint['arch'] == "vgg19":
        model = models.vgg19(pretrained = True)
    elif checkpoint['arch'] == "vgg16":
        model = models.vgg16(pretrained = True)
    
    #Freeze parameters to avoid backprograting through them
    for param in model.parameters():
        param.requires_grad = False

    model.class_to_idx = checkpoint['class_to_idx']

    # Replace classifier, loss function and optimizer
    model.classifier = nn.Sequential(nn.Linear(25088, checkpoint['hidden_units']),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(checkpoint['hidden_units'], 102),
                                     nn.LogSoftmax(dim=1))    

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr=checkpoint['learning_rate'])

    # Loading previous states
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    print("Model Loaded")
    
    return model

def process_image(image):
    """
    """
    
    image = Image.open(image)
    new_width = 244
    new_height = 244
    
    # Find the shortest side and resize to 256
    if image.size[0] > image.size[1]:
        image.thumbnail((image.size[0],256))
    else:
        image.thumbnail((256,image.size[1]))
    
    # Get the margins for a center crop
    left = (image.width - new_width)/2
    right = left + new_width
    upper = (image.height - new_height)/2
    bottom = upper + new_height
                        
    # Center crop                    
    image = image.crop((left,upper,right,bottom))
                
    # Normalize the image
    np_image = np.array(image)/255.0
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    
    # Reorder the dimensions with the color channel first
    np_image = np_image.transpose((2,0,1))
                     
    return np_image
    
def predict_cat(image_path,model,topk,cat_to_name,gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    flowers = []
    labels = []
    image = process_image(image_path)
    image = torch.from_numpy(image).unsqueeze(0).float()
    
    if gpu == True:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    
    image, model = image.to(device), model.to(device);

    # Put image through model and predict the most likely 
    logps = model.forward(image)
    ps = torch.exp(logps)
    top_p, top_class = ps.topk(topk, dim=1)
    
    top_p = top_p.squeeze().tolist()
    top_class = top_class.squeeze().tolist()
    
    # Switch the keys and values
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    
    with open(cat_to_name, 'r') as f:
        label_map = json.load(f)
    
    for x in top_class:
        labels.append(idx_to_class.get(x))
    
    for y in labels:
        flowers.append(label_map.get(y))
    
    filepath_split = image_path.split('/')
    flower_index = filepath_split[len(filepath_split)-2]
    name = label_map[flower_index]
    
    return top_p,labels,flowers,name
        