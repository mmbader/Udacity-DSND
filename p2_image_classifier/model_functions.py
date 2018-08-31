# Imports here
import torch
from torchvision import datasets, transforms, models
from collections import OrderedDict
from torch import nn
from torch import optim
import json
import torchvision
import numpy as np
import time
import datetime

def transform_load(data_dir):
    # Load the data
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    image_datasets = {'train': train_data,
                      'valid': valid_data,
                      'test': test_data}

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    data_loaders = {'train': trainloader,
                      'valid': validloader,
                      'test': testloader}

    class_names = image_datasets['train'].classes
    
    return image_datasets, data_loaders, class_names
    
def create_model(arch='densenet169', dropout=0.5, hidden_units=800, learning_rate=0.001):
    # Load a pre-trained network
    model_arch = getattr(models, arch)
    model = model_arch(pretrained=True)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    try:
        input_units = model.classifier.in_features
    except:
        try:
            input_units = model.classifier[0].in_features
        except:
            try:
                input_units = model.classifier[1].in_features
            except:
                input_units = model.fc.in_features

    # Define a new, untrained feed-forward network as classifier    
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_units, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('dropout1', nn.Dropout(dropout)),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    print("The following hyperparameters will be used:")
    print("Architecture: ", arch)
    print("Dropout: ", dropout)
    print("Hidden Units: ", hidden_units)
    print("Learning Rate: ", learning_rate)
    print("Optimization criterion is NLLLoss and optimizer is Adam")
    
    return model, criterion, optimizer

def train_model(image_datasets, data_loaders, model, criterion, optimizer, epochs, device='cuda'):
    
    print(device, "is being used for Training")
    print()
    
    model.to(device)
    total_time = 0
    
    for epoch in range(epochs):
        
        start = time.time()
        print("Epoch {}/{}".format(epoch + 1, epochs))
        print("Time: ", datetime.datetime.fromtimestamp(start).strftime('%H:%M:%S'))
        print("-----")
        
        #Training phase
        model.train()  # Set model to training mode
        running_loss = 0
        running_c_predictions = 0
        
        for inputs, labels in data_loaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            #forward
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            
            #backward
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_c_predictions += torch.sum(predicted == labels.data)
            
        epoch_loss = running_loss / len(image_datasets['train'])
        epoch_acc = running_c_predictions.double() / len(image_datasets['train'])

        print("{} Loss: {:.4f} Acc: {:.4f}".format("Training:", epoch_loss, epoch_acc))

        #Validation phase
        model.eval()  # Set model to validation mode
        running_loss = 0
        running_c_predictions = 0
        
        for inputs, labels in data_loaders['valid']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            #forward
            with torch.no_grad():
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            running_c_predictions += torch.sum(predicted == labels.data)
            
        epoch_loss = running_loss / len(image_datasets['valid'])
        epoch_acc = running_c_predictions.double() / len(image_datasets['valid'])

        print("{} Loss: {:.4f} Acc: {:.4f}".format("Validation:", epoch_loss, epoch_acc))
        
        finish = time.time()
        epoch_time = finish - start
        
        print("Epoch time:", round(epoch_time))
        print()
        
        total_time += epoch_time
        
    print("Total time:", total_time)

def model_test(image_datasets, data_loaders, model, data, criterion, device='cuda'):
    #Test phase
    model.to(device)
    model.eval()  # Set model to validation mode
    running_loss = 0
    running_c_predictions = 0

    for inputs, labels in data_loaders[data]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        #model.optim.zero_grad()

        #forward
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item()
        running_c_predictions += torch.sum(predicted == labels.data)

    epoch_loss = running_loss / len(image_datasets[data])
    epoch_acc = running_c_predictions.double() / len(image_datasets[data])

    print("{} Loss: {:.4f} Acc: {:.4f}".format("Test:", epoch_loss, epoch_acc))