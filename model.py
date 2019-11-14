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

def train(arch,learning_rate,hidden_units,epochs,gpu,dataloader,save_dir,image_datasets):
    """
    Purpose: Train the model
    Inputs: 
        Arch - String of architecture to use
        Learning Rate - Float of learning rate
        Hidden Units - Int of number of hidden units
        Epochs - Int of the number of epochs
        GPU - Binary for GPU or not
    Outputs:
        Trained Model
    """

    # Baseline all the values needed to train the classifier
    epochs = epochs
    steps = 0
    running_loss = 0
    filename = 'checkpoint.pth'
    
    #If GPU is requested by user and is possible, use GPU. If not, use CPU   
    if gpu == True:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

    # Get pretrained model
    if arch == "vgg19":
        model = models.vgg19(pretrained = True)
    elif arch == "vgg16":
        model = models.vgg16(pretrained = True)

    #Freeze parameters to avoid backprograting through them
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier, loss function and optimizer
    model.classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(hidden_units, 102),
                                     nn.LogSoftmax(dim=1))    

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr=learning_rate)

    # If applicable, move to GPU
    model.to(device);

    for e in range(epochs):
        print("Training Mode: Engaged")
        running_loss = 0
        for images, labels in dataloader['train']:
            images, labels = images.to(device), labels.to(device)

            # Train model with image - clear gradients, forward pass, get loss, back propogate & update values
            optimizer.zero_grad()
            out = model.forward(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Print out result
        else:
            print("Accuracy Check: Engaged")
            test_loss = 0
            accuracy = 0
            
            with torch.no_grad():
                model.eval()
                for inputs, labels in dataloader['test']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Test Accuracy: {:.3f}".format(accuracy/len(dataloader['test'])),
                  "Training Loss: {:.3f}.. ".format(running_loss/len(dataloader['train'])),
                  "Test Loss: {:.3f}.. ".format(test_loss/len(dataloader['test'])))
            model.train()
    
    # Validating Model
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in dataloader['validation']:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print("Validation Accuracy: {:.3f}".format(accuracy/len(dataloader['validation'])),
          "Validation Loss: {:.3f}".format(test_loss/len(dataloader['validation'])))

    # Saving the model
    model.class_to_idx = image_datasets['train'].class_to_idx
    model.cpu()

    checkpoint = {'epochs': epochs,
                  'hidden_units': hidden_units,
                  'learning_rate': learning_rate,
                  'arch': arch,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'optimizer': optimizer.state_dict()}

    torch.save(checkpoint,save_dir)
    
    return model
            
