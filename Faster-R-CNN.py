from google.colab import drive
drive.mount('/content/drive')

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(DEVICE, torch.cuda.get_device_name(0))
else:
    DEVICE = torch.device("cpu")
    print(DEVICE)

img0 = cv2.imread("/content/drive/MyDrive/zebras.jpg")
img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
print(img0.shape)
plt.imshow(img0)
plt.show()

# object information : a set of bounding boxes [x1, y1, x2, y2] 
# and their labels
bbox0 = np.array([[223, 782, 623, 1074], [597, 695, 1038, 1050], 
                  [1088, 699, 1452, 1057], [1544, 771, 1914, 1063]]) 
labels = np.array([1, 1, 1, 1]) # 0: background, 1: zebra

img0_clone = np.copy(img0)
# print(len(bbox0))
for i in range(len(bbox0)):
    cv2.rectangle(img0_clone, (bbox0[i][0], bbox0[i][1]), 
                              (bbox0[i][2], bbox0[i][3]),
                 color=(0, 255, 0), thickness=10)
plt.imshow(img0_clone)
plt.show()

img = cv2.resize(img0, dsize=(800, 800), interpolation=cv2.INTER_CUBIC)
plt.figure(figsize=(7, 7))
plt.imshow(img)
# plt.grid(True, color="black")
plt.show()

# change the bounding box coordinates

Wratio = 800/img0.shape[1]
Hratio = 800/img0.shape[0]

# print(img0.shape) -> (1333, 2000, 3)
# img0.shape[0] -> 1333
# img0.shape[1] -> 2000

# print(Wratio, Hratio)
# 0.4 0.6001500375093773


# bounding boxes [x1, y1, x2, y2] 
ratioList = [Wratio, Hratio, Wratio, Hratio]
bbox = []

# bbox0 -> [[223, 782, 623, 1074], [597, 695, 1038, 1050], 
#           [1088, 699, 1452, 1057], [1544, 771, 1914, 1063]]

# ratioList -> [0.4, 0.6001500375093773, 0.4, 0.6001500375093773]

for box in bbox0:
    box = [int(a*b) for a, b in zip(box, ratioList)]
    # print(f'box = {box}')
    # box = [89, 469, 249, 644]
    # box = [238, 417, 415, 630]
    # box = [435, 419, 580, 634]
    # box = [617, 462, 765, 637]
    bbox.append(box)

bbox = np.array(bbox)
print(bbox)

img_clone = np.copy(img)
for i in range(len(bbox)):
    cv2.rectangle(img_clone, (bbox[i][0], bbox[i][1]), (bbox[i][2], bbox[i][3]), color=(0, 255, 0), thickness=5)
plt.imshow(img_clone)
plt.show()

# only print feature extraction part of VGG16

model = torchvision.models.vgg16(pretrained=True).to(DEVICE)
features = list(model.features)
print(len(features))
print(features)

# only collect layers with output feature map size (W, H) < 50

dummy_img = torch.zeros((1, 3, 800, 800)).float() # test image array
print(dummy_img.shape)

req_features = []
output = dummy_img.clone().to(DEVICE)

for feature in features:
    output = feature(output)
    
    if output.size()[2] < 800//16: 
        # 800/16=50 이미지 size가 50보다 작으면 break
        break

    print(output.size())
    req_features.append(feature)

out_channels = output.size()[1]
# 이거 for문안에 있던거 고쳤는데 문제없을거야.

print(len(req_features))
# print(req_features)
print(out_channels)

# convert this list into a Seqeuntial module

faster_rcnn_feature_extractor = nn.Sequential(*req_features)

# print(faster_rcnn_feature_extractor)

# test the results of the input image pass through the feature extractor

transform = transforms.Compose([transforms.ToTensor()])
imgTensor = transform(img).to(DEVICE)
imgTensor = imgTensor.unsqueeze(0)
output_map = faster_rcnn_feature_extractor(imgTensor)

print(output_map.size())


# visualize the first 5 channels of the 50*50*512 feature maps

imgArray = output_map.data.cpu().numpy().squeeze(0)
fig = plt.figure(figsize=(200, 200))

start=1
end=5

print(imgArray.shape)

for i in range(start,end):
    fig.add_subplot(start, end, i)
    plt.imshow(imgArray[i], cmap='gray')
    
    
plt.show()