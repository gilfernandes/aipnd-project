#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Gil Fernandes
# DATE CREATED: 2018-12-27
# REVISED DATE:
# PURPOSE: Used to setup the neural network training model
#
#
from collections import OrderedDict
from torchvision import models
from torch import nn

model_input_size = {'vgg16': 25088, 'vgg19': 25088, 'alexnet': 9216, 'densenet161': 2208, 'resnet152': 2048}


class VisualModel():
    def __init__(self, arch='vgg16', hidden_layers=4096, drop_out=0.5, output_size=102):
        ''' Builds the actual network using a pre-defined architectures

            Arguments
            ---------
            arch: str, The CNN architecture. like e.g: alexnet, vgg16, densenet161
            hidden_layers: int, The number of hidden layers
            drop_out: float, The drop out rate
        '''
        self.arch = arch
        self.hidden_layers = hidden_layers

        self.output_size = output_size

        self.call_map = {'vgg16': self.init_vgg16, 'vgg19': self.init_vgg19, 'alexnet': self.init_alexnet,
                         'densenet161': self.init_densenet161, 'resnet152': self.init_resnet152}

        self.init_model()
        self.drop_out = drop_out
        if arch == 'resnet152':
            self.model.fc = self.create_classifier(self.hidden_layers, self.drop_out, self.output_size,
                                                   self.model.fc.in_features)
        else:
            self.model.classifier = self.create_classifier(self.hidden_layers, self.drop_out, self.output_size,
                                                           model_input_size[arch])

    def create_classifier(self, hidden_layers, drop_out, output_size, input_size):
        classifier_layers = [
            ('fc1', nn.Linear(input_size, hidden_layers)),
            ('relu1', nn.ReLU()),
            ('dropout_fc1', nn.Dropout(p=drop_out)),
            ('fc2', nn.Linear(hidden_layers, output_size)),
            ('output', nn.LogSoftmax(dim=1))
        ]
        return nn.Sequential(OrderedDict(classifier_layers))

    def init_model(self):
        self.model = self.call_map[self.arch]()

    def init_vgg16(self):
        return models.vgg16(pretrained=True)

    def init_vgg19(self):
        return models.vgg19(pretrained=True)

    def init_alexnet(self):
        return models.alexnet(pretrained=True)

    def init_densenet161(self):
        return models.densenet161(pretrained=True)

    def init_resnet152(self):
        return models.resnet152(pretrained=True)

    def input_size(self):
        return model_input_size[self.arch]


if __name__ == "__main__":
    visual_model = VisualModel('alexnet', hidden_layers=512)
    print(visual_model.model)
    visual_model = VisualModel('densenet161', hidden_layers=512)
    print(visual_model.model)
    visual_model = VisualModel('vgg16', hidden_layers=4096)
    print(visual_model.model)
    visual_model = VisualModel('vgg19', hidden_layers=4096)
    print(visual_model.model)
