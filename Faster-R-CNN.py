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

# sub-sampling rate = 1/16
# image size : 800x800
# sub-sampled feature map size : 800 x 1/16 = 50
# 50 x 50 = 2500 anchors and each anchor generate 9 anchor boxes
# total anchor boxes = 50 x 50 x 9 = 22500
# x,y intervals to generate anchor box center

feature_size = 800 // 16
ctr_x = np.arange(16, (feature_size + 1) * 16, 16)
ctr_y = np.arange(16, (feature_size + 1) * 16, 16)
print(len(ctr_x))
print(ctr_x)

# coordinates of the 255 center points to generate anchor boxes

index = 0
ctr = np.zeros((2500, 2))

for i in range(len(ctr_x)):
    for j in range(len(ctr_y)):
        ctr[index, 1] = ctr_x[i] - 8
        ctr[index, 0] = ctr_y[j] - 8
        index += 1

# ctr => [[center x, center y], ...]
print(ctr.shape)
print(ctr[:10, :])

# display the 2500 anchors within image

img_clone2 = np.copy(img)
ctr_int = ctr.astype("int32")

plt.figure(figsize=(7, 7))
for i in range(ctr.shape[0]):
    cv2.circle(img_clone2, (ctr_int[i][0], ctr_int[i][1]),
              radius=1, color=(255, 0, 0), thickness=3)
plt.imshow(img_clone2)
plt.show()

# for each of the 2500 anchors, generate 9 anchor boxes
# 2500 x 9 = 22500 anchor boxes

ratios = [0.5, 1, 2]
scales = [8, 16, 32]
sub_sample = 16

anchor_boxes = np.zeros(((feature_size * feature_size * 9), 4))
# anchor_boxes = (50*50*9,4)

index = 0

# ctr -> (2500,2) 
# (8,8), (8,24), (8,40) ....784 , 784

for c in ctr:                        # per anchors
    ctr_y, ctr_x = c
    for i in range(len(ratios)):     # per ratios
        for j in range(len(scales)): # per scales
            
            # anchor box height, width
            h = sub_sample * scales[j] * np.sqrt(ratios[i])
            w = sub_sample * scales[j] * np.sqrt(1./ ratios[i])
            # np.sqrt 빼고 생각하면 편함
            # h, w 자체가 bounding box를 생성하기위한 목적!

            # anchor box [x1, y1, x2, y2]
            anchor_boxes[index, 0] = ctr_x - w / 2.
            anchor_boxes[index, 1] = ctr_y - h / 2.
            anchor_boxes[index, 2] = ctr_x + w / 2.
            anchor_boxes[index, 3] = ctr_y + h / 2.
            
            index += 1
            
print(anchor_boxes.shape)
print(anchor_boxes[:10, :])

# display the anchor boxes of one anchor and the ground truth boxes

img_clone = np.copy(img)
# 만약 new = arr 이렇게 복사할시
# new[0] = 1 로 바꾼다면 arr 까지 바뀌기때문에 np.copy 를 해줘야한다 np.copy 시 본래의 배열에 지장을 주지않음.

# draw random anchor boxes
# 22500 / 2 = 11250
for i in range(11250, 11259):
    x1 = int(anchor_boxes[i][0])
    y1 = int(anchor_boxes[i][1])
    x2 = int(anchor_boxes[i][2])
    y2 = int(anchor_boxes[i][3])
    
    cv2.rectangle(img_clone, (x1, y1), (x2, y2), color=(255, 0, 0),
                 thickness=3)

# draw ground truth boxes
for i in range(len(bbox)):
    cv2.rectangle(img_clone, (bbox[i][0], bbox[i][1]), 
                             (bbox[i][2], bbox[i][3]),
                 color=(0, 255, 0), thickness=3)

plt.imshow(img_clone)
plt.show()


# draw all anchor boxes

# add paddings(can't draw anchor boxes out of image boundary)
img_clone3 = np.copy(img)
img_clone4 = cv2.copyMakeBorder(img_clone3,400,400,400,400,cv2.BORDER_CONSTANT, value=(255, 255, 255))
img_clone5 = np.copy(img_clone4)

for i in range(len(anchor_boxes)):
    x1 = int(anchor_boxes[i][0])
    y1 = int(anchor_boxes[i][1])
    x2 = int(anchor_boxes[i][2])
    y2 = int(anchor_boxes[i][3])
    
    cv2.rectangle(img_clone5, (x1+400, y1+400), (x2+400, y2+400), color=(255, 0, 0),
                 thickness=2)

plt.figure(figsize=(10, 10))
plt.subplot(121), plt.imshow(img_clone4)
plt.subplot(122), plt.imshow(img_clone5)
plt.show()


# ignore the cross-boundary anchor boxes
# valid anchor boxes with (x1, y1) > 0 and (x2, y2) <= 800

index_inside = np.where(
        (anchor_boxes[:, 0] >= 0) &
        (anchor_boxes[:, 1] >= 0) &
        (anchor_boxes[:, 2] <= 800) &
        (anchor_boxes[:, 3] <= 800))[0]


# [0]을 하는이유는 where로 나오면 numpy array 형태로 나오게되는데, 배열 인덱스값을 빼내기위해

print(index_inside.shape)

# only 8940 anchor boxes are inside the boundary out of 22500
valid_anchor_boxes = anchor_boxes[index_inside]
print(valid_anchor_boxes.shape)





