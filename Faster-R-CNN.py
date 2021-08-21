import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import PIL

img = PIL.Image.open('./zebra.jpg')

resize = transforms.Resize((800,800))
totensor = transforms.ToTensor()

img = resize(img)
imgTensor = totensor(img)
imgTensor = imgTensor.unsqueeze(0)
print(imgTensor.shpae)

# device = torch.device('cuda')
# model = torchvision.models.vgg16(pretrained=True).to(device)
# features = list(model.features)

# # only collect layers with output feature map size (W, H) < 50
# dummy_img = torch.zeros((1, 3, 800, 800)).float() # test image array

# req_features = []
# output = dummy_img.clone().to(device)

# for feature in features:
#     output = feature(output)
# #     print(output.size()) => torch.Size([batch_size, channel, width, height])
#     if output.size()[2] < 800//16: # 800/16=50
#         break
#     req_features.append(feature)
#     out_channels = output.size()[1]

# faster_rcnn_feature_extractor = nn.Sequential(*req_features)

# output_map = faster_rcnn_feature_extractor(imgTensor)