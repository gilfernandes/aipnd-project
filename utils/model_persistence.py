#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Gil Fernandes
# DATE CREATED: 2018-12-27
# REVISED DATE:
# PURPOSE: Used to save and load the neural network model from files
#
#
import torch

from train_model import VisualModel


class ModelPersistence():

    def __init__(self):
        """
        Contains methods for saving the model checkpoint and also loading checkpoints from disk
        """

    def save_model(self, class_to_idx_map, visual_model, file='checkpoint.pth'):
        """
        Saves the model to a file
        :param class_to_idx_map: dictionary
        :type file: str
        """
        checkpoint = {
            'output_size': visual_model.output_size,
            'hidden_layers': visual_model.hidden_layers,
            'state_dict': visual_model.model.state_dict(),
            'drop_out': visual_model.drop_out,
            'arch': visual_model.arch,
            'class_to_idx_map': class_to_idx_map
        }
        torch.save(checkpoint, file)

    def load_model(self, gpu, file='checkpoint.pth'):
        checkpoint = torch.load(file)
        hidden_layers = checkpoint['hidden_layers']
        output_size = checkpoint['output_size']
        drop_out = checkpoint.get('drop_out', 0.5)
        arch = checkpoint['arch']
        class_to_idx_map = checkpoint['class_to_idx_map']
        new_visual_model = VisualModel(arch, hidden_layers, drop_out, output_size)
        model = new_visual_model.model
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = class_to_idx_map
        model.idx_to_class = {v: k for k, v in model.class_to_idx.items()}
        model.eval()
        if gpu:
            return model.to('cuda')
        return model


