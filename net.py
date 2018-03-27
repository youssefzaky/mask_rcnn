import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models.resnet import Bottleneck
import numpy as np

import pickle
import math


class BackBone(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3]):
        self.inplanes = 64
        super(BackBone, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # differences between pytorch and caffe2 resnet
        self.layer2[0].conv1.stride = (2, 2)
        self.layer2[0].conv2.stride = (1, 1)
        self.layer3[0].conv1.stride = (2, 2)
        self.layer3[0].conv2.stride = (1, 1)
        self.layer4[0].conv1.stride = (2, 2)
        self.layer4[0].conv2.stride = (1, 1)

        self.avgpool = nn.AvgPool2d(7)
        self.top = nn.Sequential(self.layer4, self.avgpool)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # forward only to layer 3
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)


class RPN(nn.Module):
    def __init__(self, in_channels, out_channels, n_anchors, proposal_layer):
        super(RPN, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)  # first convolutional layer
        self.bbox_pred = nn.Conv2d(out_channels, n_anchors * 4, 1)  # bbox convolutional layer
        self.cls_score = nn.Conv2d(out_channels, n_anchors, 1)  # score convolutional layer
        self.proposal_layer = proposal_layer

    def forward(self, feature_map, h, w, im_scale):
        x = self.conv(feature_map)
        x = F.relu(x, inplace=True)
        cls_score = self.cls_score(x)
        cls_prob = F.sigmoid(cls_score)  # sigmoid activation, object vs no object
        bbox_pred = self.bbox_pred(x)
        proposals, scores = self.proposal_layer(cls_prob, bbox_pred,
                                                h, w, im_scale)
        return proposals, scores


class FasterRCNN(nn.Module):
    def __init__(self, rpn, backbone, roi_align, n_classes):
        super(FasterRCNN, self).__init__()
        self.backbone = backbone  # eg. Resnet-50
        self.rpn = rpn  # Region Proposal Network
        self.roi_align = roi_align  # ROI-Align layer
        self.bbox_pred = nn.Linear(2048, 4 * n_classes)  # bounding-box head
        self.cls_score = nn.Linear(2048, n_classes)  # class logits head

    def forward(self, images, h, w, im_scale):
        features = self.backbone(images)  # compute block C4 features for image
        proposals, scores = self.rpn(features, h, w, im_scale)  # apply RPN to get proposals
        pooled_feat = self.roi_align(features, proposals)  # apply ROI align on the C4 features using the proposals
        pooled_feat = self.backbone.top(pooled_feat).squeeze()  # apply the C5 block and Average pool of Resnet
        bbox_pred = self.bbox_pred(pooled_feat)  # apply bounding-box head
        cls_score = self.cls_score(pooled_feat)  # apply class-score head
        cls_prob = F.softmax(cls_score, dim=1)  # softmax to get object-class probabilities

        return {'bbox_pred': bbox_pred, 'cls_prob': cls_prob, 'rois': proposals}

    def load_pretrained_weights(self, caffe_pkl_file, mapping_file):
        with open(caffe_pkl_file, 'rb') as f:
            caffe_data = pickle.load(f, encoding='latin1')
        caffe_data = caffe_data['blobs']
        mapping = np.load(mapping_file)
        model_dict = self.state_dict()
        for i in range(len(mapping)):
            pytorch_key = 'backbone.' + mapping[i][0]
            caffe2_key = mapping[i][1]
            if model_dict[pytorch_key].size() == torch.FloatTensor(caffe_data[caffe2_key]).size():
                if i == 0:  # convert from BGR to RGB
                    model_dict[pytorch_key] = torch.FloatTensor(caffe_data[caffe2_key][:, (2, 1, 0), :, :])
                else:
                    model_dict[pytorch_key] = torch.FloatTensor(caffe_data[caffe2_key])
            else:
                print(str(i) + ',' + pytorch_key + ',' + caffe2_key)
                raise ValueError('size mistmatch')
        # load faster-RCNN weights
        self.load_state_dict(model_dict)
        self.bbox_pred.weight.data = torch.FloatTensor(caffe_data['bbox_pred_w'])
        self.bbox_pred.bias.data = torch.FloatTensor(caffe_data['bbox_pred_b'])
        self.cls_score.weight.data = torch.FloatTensor(caffe_data['cls_score_w'])
        self.cls_score.bias.data = torch.FloatTensor(caffe_data['cls_score_b'])
        # load RPN weights
        self.rpn.conv.weight.data = torch.FloatTensor(caffe_data['conv_rpn_w'])
        self.rpn.conv.bias.data = torch.FloatTensor(caffe_data['conv_rpn_b'])
        self.rpn.cls_score.weight.data = torch.FloatTensor(caffe_data['rpn_cls_logits_w'])
        self.rpn.cls_score.bias.data = torch.FloatTensor(caffe_data['rpn_cls_logits_b'])
        self.rpn.bbox_pred.weight.data = torch.FloatTensor(caffe_data['rpn_bbox_pred_w'])
        self.rpn.bbox_pred.bias.data = torch.FloatTensor(caffe_data['rpn_bbox_pred_b'])