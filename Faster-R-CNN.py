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
        # 800/16=50 ????????? size??? 50?????? ????????? break
        break

    print(output.size())
    req_features.append(feature)

out_channels = output.size()[1]
# ?????? for????????? ????????? ???????????? ??????????????????.

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
            # np.sqrt ?????? ???????????? ??????
            # h, w ????????? bounding box??? ?????????????????? ??????!

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
# ?????? new = arr ????????? ????????????
# new[0] = 1 ??? ???????????? arr ?????? ?????????????????? np.copy ??? ??????????????? np.copy ??? ????????? ????????? ????????? ????????????.

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


# [0]??? ??????????????? where??? ????????? numpy array ????????? ??????????????????, ?????? ??????????????? ???????????????

print(index_inside.shape)

# only 8940 anchor boxes are inside the boundary out of 22500
valid_anchor_boxes = anchor_boxes[index_inside]
print(valid_anchor_boxes.shape)

# calculate Iou of the valid anchor boxes
# since we have 8940 anchor boxes and 4 ground truth objects,
# we should get an array with (8940, 4) as the output
# [IoU with gt box1, IoU with gt box2, IoU with gt box3,IoU with gt box4]

ious = np.empty((len(valid_anchor_boxes),4), dtype=np.float32)
# iou??? ????????? ????????? ????????? float32 ??? ????????????
# valid_anchor_boxes -> inside image anchor box
# empty -> ????????? ????????? ????????? 0?????? ???????????? ??????????????? ?????? ????????? ???????????????.
ious.fill(0)

# anchor boxes
for i, anchor_box in enumerate(valid_anchor_boxes):
    xa1, ya1, xa2, ya2 = anchor_box
    anchor_area = (xa2 - xa1) * (ya2 - ya1)
    # size of box -> anchor_area
    
    # ground truth boxes
    # bbox -> ground truth (4,4)
    for j, gt_box in enumerate(bbox):
        xb1, yb1, xb2, yb2 = gt_box
        box_area = (xb2 - xb1) * (yb2 - yb1)

        inter_x1 = max([xb1, xa1])
        inter_y1 = max([yb1, ya1])
        inter_x2 = min([xb2, xa2])
        inter_y2 = min([yb2, ya2])
        
        if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            iou = inter_area / (anchor_area + box_area - inter_area)
        else:
            iou = 0

        # i -> anchor box index j -> ground truth index

        # i -> 0~8939 j -> 0~3
        ious[i, j] = iou

print(ious.shape)

# ????????? ??? 0, 1, 2, 3 ??? ????????????????
# ??????
# i : anchor box index 
# j : ground truth index
# ????????? 8940 anchor box index height??? ?????? 4?????? ground truth 
# ????????? ???????????? ???????????????.

print(ious[6000:6100, :])

# what anchor box has max ou with the ground truth box

gt_argmax_ious = ious.argmax(axis=0)
print(gt_argmax_ious)
# argmax -> ?????? ???????????? ????????????
# ground truth ??? ?????????????????? argmax ??? ??????. (axis = 0)

gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
print(gt_max_ious)

gt_argmax_ious = np.where(ious == gt_max_ious)[0]
print(gt_argmax_ious)
# gt_argmax_ious ??? ????????? ????????? ??????????????????, ?????? argmax ?????? ?????????????????? ?????? ????????????.
# np.where ?????? [0]??? ??????????????? index?????? ???????????? ?????????(y??????)?????? ?????? ??????????????? ????????????. ([1]??? ?????????) 



# After leaving only max IoUs
# img_clone = np.copy(img)

# # draw random anchor boxes
# # 22500 / 2 = 11250
# for i in gt_argmax_ious:
#     x1 = int(valid_anchor_boxes[i][0])
#     y1 = int(valid_anchor_boxes[i][1])
#     x2 = int(valid_anchor_boxes[i][2])
#     y2 = int(valid_anchor_boxes[i][3])
    
#     cv2.rectangle(img_clone, (x1, y1), (x2, y2), color=(255, 0, 0),
#                  thickness=3)

# # draw ground truth boxes
# for i in range(len(bbox)):
#     cv2.rectangle(img_clone, (bbox[i][0], bbox[i][1]), 
#                              (bbox[i][2], bbox[i][3]),
#                  color=(0, 255, 0), thickness=3)

# plt.imshow(img_clone)
# plt.show()

# what ground truth bbox is associated with each anchor box

argmax_ious = ious.argmax(axis=1)
print(argmax_ious.shape)
print(argmax_ious)

max_ious = ious[np.arange(len(index_inside)), argmax_ious]
# ?????? arrange??? 0~8940 ?????? ????????? ????????????, argmax_ious ??? ious??? ???????????? ????????? ???????????? ???????????? (8940,0)??? ????????? ??????
print(max_ious)


# set the labels of 8940 valid anchor boxes to -1(ignore)
label = np.empty((len(index_inside),), dtype=np.int32)
label.fill(-1)
print(label.shape)


# use IoU to assign 1 (objects) to two kind of anchors
# a) the anchors with the highest IoU overlap with a ground truth box
# b) an anchor that has an IoU overlap higher than 0.7 with ground truth box

# Assign 0 (background) to an anchor if its IoU ratio is lower than 0.3

pos_iou_threshold = 0.7
neg_iou_threshold = 0.3

label[gt_argmax_ious] = 1
label[max_ious >= pos_iou_threshold] = 1
label[max_ious < neg_iou_threshold] = 0

# print(label[8100:8200])

# Every time mini-batch training take only 256 valid anchor boxes to train RPN
# of which 128 positive examples, 128 negative-examples
# disable leftover positive/negative anchors

n_sample = 256
pos_ratio = 0.5
n_pos = pos_ratio * n_sample

pos_index = np.where(label == 1)[0]
# ????????? label[gt_argmax_ious] = 1 ??? ??????????????? iou_threshold??? 0.7??? ???????????? ?????????
# print(pos_index)

if len(pos_index) > n_pos:
    disable_index = np.random.choice(pos_index,
                                    size = (len(pos_index) - n_pos),
                                    replace=False)
    label[disable_index] = -1
    
n_neg = n_sample * np.sum(label == 1)
neg_index = np.where(label == 0)[0]

if len(neg_index) > n_neg:
    disable_index = np.random.choice(neg_index, 
                                    size = (len(neg_index) - n_neg), 
                                    replace = False)
    label[disable_index] = -1

    # convert the format of valid anchor boxes [x1, y1, x2, y2] For each valid anchor box, find the groundtruth object which has max_iou

# bbox -> ground truth boxes
max_iou_bbox = bbox[argmax_ious]
# argmax_ious -> 0 0 0 3 0 2 0 1 ??? ious ?????? ?????? ?????? ground truth index 
print(max_iou_bbox.shape)

# valid_anchor_boxes -> (x1,y1,x2,y2)
height = valid_anchor_boxes[:, 3] - valid_anchor_boxes[:, 1]
width = valid_anchor_boxes[:, 2] - valid_anchor_boxes[:, 0]

ctr_y = valid_anchor_boxes[:, 1] + 0.5 * height
ctr_x = valid_anchor_boxes[:, 0] + 0.5 * width

# ctr_x = 1/2*width 
# ctr_y -> 1/2*height ???????????? height width

# [x1,y1,x2,y2]
base_height = max_iou_bbox[:, 3] - max_iou_bbox[:, 1]
base_width = max_iou_bbox[:, 2] - max_iou_bbox[:, 0]

base_ctr_y = max_iou_bbox[:, 1] + 0.5 * base_height
base_ctr_x = max_iou_bbox[:, 0] + 0.5 * base_width
# ?????? 

eps = np.finfo(height.dtype).eps
# eps -> ???????????????

height = np.maximum(height, eps)
width = np.maximum(width, eps)
# ???????????? ??????.
############################################################
############################################################
############################################################
# ????????? ?????? ???????????? ?????????....!
dy = (base_ctr_y - ctr_y) / height
dx = (base_ctr_x - ctr_x) / width
dh = np.log(base_height / height)
dw = np.log(base_width / width)

anchor_locs = np.vstack((dx, dy, dw, dh)).transpose()
print(anchor_locs.shape)
# ????????? ?????? ???????????? ?????????....!
############################################################
############################################################
############################################################

# First set the label=-1 and locations=0 of the 22500 anchor boxes, 
# and then fill in the locations and labels of the 8940 valid anchor boxes
# NOTICE: For each training epoch, we randomly select 128 positive + 128 negative 
# from 8940 valid anchor boxes, and the others are marked with -1

anchor_labels = np.empty((len(anchor_boxes)), dtype=label.dtype)
anchor_labels.fill(-1)

# label[gt_argmax_ious] = 1
# label[max_ious >= pos_iou_threshold] = 1
# label[max_ious < neg_iou_threshold] = 0
anchor_labels[index_inside] = label
# index_inside -> ????????? ??????????????? anchor box index
# max_IoU ??????????????? 1??? ?????????????????????    
# if anchor label == 0  negative label, 
# anchor label == 1 positive label, 
# anchor label == -1  nothing
# print(anchor_labels[8700:8720])
print(anchor_labels.shape)

anchor_locations = np.empty((len(anchor_boxes),) + anchor_boxes.shape[1:], dtype=anchor_locs.dtype)
anchor_locations.fill(0)
anchor_locations[index_inside, :] = anchor_locs
print(anchor_locations.shape)
print(anchor_locations[:10, :])


# Send the features of the input image to the Region Proposal Network (RPN), 
# predict 22500 region proposals (ROIs)

in_channels = 512
mid_channels = 512
n_anchor = 9

conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1).to(DEVICE)
conv1.weight.data.normal_(0, 0.01)
conv1.bias.data.zero_()

# bounding box regressor
reg_layer = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0).to(DEVICE)
reg_layer.weight.data.normal_(0, 0.01)
reg_layer.bias.data.zero_()

# classifier(object or not)
cls_layer = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0).to(DEVICE)
cls_layer.weight.data.normal_(0, 0.01)
cls_layer.bias.data.zero_()


x = conv1(output_map.to(DEVICE)) # output_map = faster_rcnn_feature_extractor(imgTensor)
pred_anchor_locs = reg_layer(x) # bounding box regresor output
pred_cls_scores = cls_layer(x)  # classifier output 

print(pred_anchor_locs.shape, pred_cls_scores.shape)


# Convert RPN to predict the position and classification format of the anchor box
# Position: [1, 36(9*4), 50, 50] => [1, 22500(50*50*9), 4] (dy, dx, dh, dw) 
# Classification: [1, 18(9*2), 50, 50] => [1, 22500, 2] (1, 0)

pred_anchor_locs = pred_anchor_locs.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)
print(pred_anchor_locs.shape)

pred_cls_scores = pred_cls_scores.permute(0, 2, 3, 1).contiguous()
print(pred_cls_scores.shape)

objectness_score = pred_cls_scores.view(1, 50, 50, 9, 2)[:, :, :, :, 1].contiguous().view(1, -1)
print(objectness_score.shape)

pred_cls_scores = pred_cls_scores.view(1, -1, 2)
print(pred_cls_scores.shape)

# According to the 22500 ROIs predicted by RPN and 22500 anchor boxes, 
# calculate the RPN loss??
print(pred_anchor_locs.shape)
print(pred_cls_scores.shape)
print(anchor_locations.shape)
print(anchor_labels.shape)