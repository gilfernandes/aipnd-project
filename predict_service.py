#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Gil Fernandes
# DATE CREATED: 2018-12-27
# REVISED DATE:
# PURPOSE: Contains methods for transforming the image to the format known by Pytorch and
# for creating the actual prediction
#
#
from PIL import Image
import numpy as np
import torch

from utils.cat_to_name import CategoryNameMapping


class PredictService():
    def __init__(self, output_size) -> None:
        self.output_size = output_size

    def process_image(self, image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''

        im = Image.open(image)
        width = 256
        height = 256
        size = width, height
        im.thumbnail(size)
        new_width = 224
        new_height = 224
        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2
        im = im.crop((left, top, right, bottom))
        np_image = np.array(im, dtype='float64')
        np_image /= np.array([255, 255, 255])
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        np_image = (np_image - mean) / std
        return torch.from_numpy(np_image.transpose((2, 0, 1)))

    def predict(self, image_path, model, gpu=False, topk=5):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''

        processed_image = self.process_image(image_path)
        if gpu:
            processed_image = processed_image.to('cuda')
        processed_image.resize_([1, 3, 224, 224])
        outputs = model(processed_image.float())
        probs, classes = (torch.exp(outputs.resize(self.output_size)).topk(topk))
        idx = [model.idx_to_class[x.item()] for x in classes]
        probs = [x.item() for x in probs]
        return probs, idx

    def map_to_category(self, category_names, idx):
        mapping = CategoryNameMapping(category_names)
        return [f"{mapping.cat_to_name.get(x, 'Unknown')} ({x})" for x in idx]
