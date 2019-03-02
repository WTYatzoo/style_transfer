import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }
        for p in self.parameters():
            p.requires_grad = False
    
    def forward(self, x):
        outs = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                outs.append(x)
        return outs


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_transpose = features.transpose(1, 2)
    gram = features.bmm(features_transpose) / (ch * h * w)
    return gram

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg16 = models.vgg16(pretrained=True)
vgg16 = VGG(vgg16.features[:23]).to(device).eval()

content_img=cv2.imread("./test_1/content.png")
style_img=cv2.imread("./test_1/style_4.png")
shape_content_img=content_img.shape
shape_style_img=style_img.shape
content_img=content_img.reshape(1,shape_content_img[0],shape_content_img[1],shape_content_img[2])
style_img=style_img.reshape(1,shape_style_img[0],shape_style_img[1],shape_style_img[2])

content_img=torch.Tensor(content_img)
style_img=torch.Tensor(style_img)
content_img=content_img.permute(0,3,1,2)
style_img=style_img.permute(0,3,1,2)

print(content_img.shape)
print(style_img.shape)
input_img = content_img.clone()
content_features = vgg16(content_img.cuda())
style_features=vgg16(style_img.cuda())
style_grams = [gram_matrix(x) for x in style_features]

optimizer = optim.LBFGS([input_img.requires_grad_()])
style_weight = 1e3
content_weight = 1

run = 0
while run <= 600:
    def f():
        global run
        optimizer.zero_grad()
        features = vgg16(input_img.cuda())        
        content_loss = F.mse_loss(features[2], content_features[2]) * content_weight
        style_loss = 0
        grams = [gram_matrix(x) for x in features]
        for a, b in zip(grams, style_grams):
            style_loss += F.mse_loss(a, b) * style_weight
        
        loss = style_loss + content_loss
        
        if run % 50 == 0:
            print('Step {}: Style Loss: {:4f} Content Loss: {:4f}'.format(run, style_loss.item(), content_loss.item()))
        run += 1
        
        loss.backward()
        return loss
    
    optimizer.step(f)

input_img=input_img.permute(0,2,3,1)
input_img=input_img.detach().numpy()
input_img=input_img.reshape(shape_content_img[0],shape_content_img[1],shape_content_img[2])
cv2.imwrite("./test_1/final_4.png",input_img)
    
