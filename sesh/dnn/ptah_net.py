import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import sesh.dnn as model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms

num_channels = 3
model_dir = '../sesh/models/'
model_load_path = model_dir+'ptah.pth'

trained_signs = np.load(model_dir+'class_names.npy')
trained_signs.sort()

num_signs = len(trained_signs)

class PtahNet(nn.Module):
    def __init__(self):
        super(PtahNet, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 1000)
        self.fc2 = nn.Linear(1000, 128)
        self.fc3 = nn.Linear(128, num_signs)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))   
        x = x.view(-1, 32 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize([32, 32]),        
        transforms.Grayscale(num_channels),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.Grayscale(num_channels),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),    
    'ocr': transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.Grayscale(num_channels),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ,]),
    'ocr2': transforms.Compose([
        transforms.Resize([32, 32])
    ]), 
}

def get_sign_from_image(img):
    model = PtahNet()

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_load_path))
    else:
        model.load_state_dict(torch.load(model_load_path, map_location=torch.device('cpu')))

    model.eval()

    with torch.no_grad():
        img_tensor = data_transforms['ocr'](img)
        img_tensor.unsqueeze_(0)

        outputs = model(img_tensor)
        _, preds = torch.max(outputs,1)      

    return trained_signs[preds[0]]
