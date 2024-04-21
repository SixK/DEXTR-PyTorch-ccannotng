#!/usr/bin/env python

import os
import sys
import torch
from collections import OrderedDict
from PIL import Image
import numpy as np

# from torch.nn.functional import upsample
from torch.nn.functional import interpolate

import networks.deeplab_resnet as resnet
from dataloaders import helpers as helpers

gpu_id = 0
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")

class DEXTR(object):
    """Pyramid Scene Parsing Network by Hengshuang Zhao et al 2017"""

    def __init__(self, nb_classes, resnet_layers, input_shape, num_input_channels=4,
                 classifier='psp', weights_path='models/dextr_pascal-sbd.pth', sigmoid=False):
        self.input_shape=input_shape
        net = resnet.resnet101(1, nInputChannels=4, classifier='psp')
        print("Initializing weights from: {}".format(weights_path))
        state_dict_checkpoint = torch.load(weights_path,
                                           map_location=lambda storage, loc: storage)
        # Remove the prefix .module from the model when it is trained using DataParallel
        if 'module.' in list(state_dict_checkpoint.keys())[0]:
            new_state_dict = OrderedDict()
            for k, v in state_dict_checkpoint.items():
                name = k[7:]  # remove `module.` from multi-gpu training
                new_state_dict[name] = v
        else:
            new_state_dict = state_dict_checkpoint
        net.load_state_dict(new_state_dict)
        net.eval()
        net.to(device)
        self.net=net
    
    def predict_mask(self, image, points, pad=50, threshold=0.8, zero_pad=True):
        points = np.array(points).astype(int)
        image = np.array(image)

        #  Crop image to the bounding box from the extreme points and resize
        bbox = helpers.get_bbox(image, points=points, pad=pad, zero_pad=True)
        crop_image = helpers.crop_from_bbox(image, bbox, zero_pad=True)
        resize_image = helpers.fixed_resize(crop_image, (512, 512)).astype(np.float32)

        #  Generate extreme point heat map normalized to image values
        extreme_points = points - [np.min(points[:, 0]), np.min(points[:, 1])] + [pad, pad]
        extreme_points = (512 * extreme_points * [1 / crop_image.shape[1], 1 / crop_image.shape[0]]).astype(int)
        extreme_heatmap = helpers.make_gt(resize_image, extreme_points, sigma=10)
        extreme_heatmap = helpers.cstm_normalize(extreme_heatmap, 255)

        #  Concatenate inputs and convert to tensor
        input_dextr = np.concatenate((resize_image, extreme_heatmap[:, :, np.newaxis]), axis=2)
        inputs = torch.from_numpy(input_dextr.transpose((2, 0, 1))[np.newaxis, ...])

        # Run a forward pass
        inputs = inputs.to(device)
        outputs = self.net.forward(inputs)
        # outputs = upsample(outputs, size=(512, 512), mode='bilinear', align_corners=True)
        outputs = interpolate(outputs, size=(512, 512), mode='bilinear', align_corners=True)
        outputs = outputs.to(torch.device('cpu'))

        pred = np.transpose(outputs.data.numpy()[0, ...], (1, 2, 0))
        pred = 1 / (1 + np.exp(-pred))
        pred = np.squeeze(pred)
        result = helpers.crop2fullmask(pred, bbox, im_size=image.shape[:2], zero_pad=True, relax=pad) > threshold

        return result
