import torch
import numpy as np
from torchvision import models
from torch import nn
from collections import OrderedDict
from torch import optim
import json
import matplotlib.pyplot as plt

def save_model(image_datasets, arch, model, dropout, hidden_units, learning_rate, epochs, optimizer, checkpoint='checkpoint.pth'):
    # TODO: Save the checkpoint
    model.class_to_idx = image_datasets['train'].class_to_idx
    torch.save({'arch': arch,
                'model_state_dict': model.state_dict(),
                'dropout': dropout,
                'hidden_units': hidden_units,
                'learning_rate': learning_rate,
                'epochs': epochs,
                'opt_state_dict': optimizer.state_dict(),
                'class_to_idx': model.class_to_idx},
                checkpoint)
    
# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath, device):
    checkpoint = torch.load(filepath, map_location=str(device))

    # Load a pre-trained network
    arch = checkpoint['arch']
    model_arch = getattr(models, arch)
    model = model_arch(pretrained=True)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    # Set hyperparameters
    dropout = checkpoint['dropout']
    input_units = model.classifier.in_features
    hidden_units = checkpoint['hidden_units']
    learning_rate = checkpoint['learning_rate']
    epochs = checkpoint['epochs']

    # Define a new, untrained feed-forward network as classifier    
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_units, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('dropout1', nn.Dropout(dropout)),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    
    model.classifier = classifier
    
    model.class_to_idx = checkpoint['class_to_idx']

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['opt_state_dict'])
    
    return model, optimizer, criterion

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    from PIL import Image
    pil_img = Image.open(image_path)
    
    if pil_img.size[0] > pil_img.size[1]:
        pil_img.thumbnail((20000, 256))
    else:
        pil_img.thumbnail((256, 30000))
         
    l_margin = (pil_img.width-224)/2
    b_margin = (pil_img.height-224)/2
    r_margin = l_margin + 224
    t_margin = b_margin + 224
    pil_img = pil_img.crop((l_margin, b_margin, r_margin, t_margin))

    pil_img = np.array(pil_img)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    pil_img = (pil_img - mean)/std
    
    pil_img = pil_img.transpose((2, 0, 1))
    
    return pil_img

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
        
    if title:
        plt.title(title)
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(cat_to_name, image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    model.to(device)
    img = process_image(image_path)

    #if device=='cpu':
    #    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    #else:
    #    image_tensor = torch.from_numpy(img).type(torch.FloatTensor).cuda()
    
    if device == 'cpu':
        image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    else:
        image_tensor = torch.from_numpy(img).type(torch.cuda.FloatTensor)
    print('Predicting flower class using:', device)

    model_input = image_tensor.unsqueeze(0)
    
    probs = torch.exp(model.forward(model_input))
    
    top_probs, top_labs = probs.topk(topk)
    top_probs = top_probs.detach().cpu().numpy().tolist()[0] 
    top_labs = top_labs.detach().cpu().numpy().tolist()[0]
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]
    
    flower_num = image_path.split('/')[2]
    title_ = cat_to_name[flower_num]
    
    rounded_probs = [round(elem, 2) for elem in top_probs]
    
    print("Image is showing:",title_)
    print("Probabilities:",rounded_probs)
    print("Labels:",top_labels)
    print("Flowers:",top_flowers)

def label_map(label_file='cat_to_name.json'):
    # Label mapping
    with open(label_file, 'r') as f:
        cat_to_name = json.load(f)
    
    return cat_to_name