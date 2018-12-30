#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Gil Fernandes
# DATE CREATED: 2018-12-27
# REVISED DATE:
# PURPOSE: Used to setup an object with all the variables needed for data loading
#
#
import torch
from torchvision import datasets, transforms


class DataLoader():
    def __init__(self, data_dir, batch_size=16, image_size=224):
        ''' Builds a feedforward network with arbitrary hidden layers.

            Arguments
            ---------
            data_dir: string, the directory where the test data is found
            cat_to_name_file: The category to name mapping json file
            batch_size: integer, size of the batch to be read each time by the loader
        '''
        self.data_dir = data_dir
        self.train_dir = data_dir + '/train'
        self.valid_dir = data_dir + '/valid'
        self.test_dir = data_dir + '/test'

        normalize_means = [0.485, 0.456, 0.406]
        normalize_stds = [0.485, 0.456, 0.406]
        self.train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                                    transforms.RandomResizedCrop(image_size),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(normalize_means,
                                                                         normalize_stds)])

        self.test_transforms = transforms.Compose([transforms.Resize(256),
                                                   transforms.CenterCrop(image_size),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(normalize_means,
                                                                        normalize_stds)])

        self.validation_transforms = self.test_transforms

        # Load the datasets with ImageFolder
        self.train_dataset = datasets.ImageFolder(self.train_dir, transform=self.train_transforms)
        test_dataset = datasets.ImageFolder(self.test_dir, transform=self.test_transforms)
        validation_dataset = datasets.ImageFolder(self.valid_dir, transform=self.validation_transforms)

        # Using the image datasets and the trainforms, define the dataloaders
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
        self.validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size)


if __name__ == "__main__":
    data_loader = DataLoader('/dev/playground/ai_programming_udacity/data/flowers', 'cat_to_name.json', 32)
    print('train_dir', data_loader.train_dir)
    print('train_dataloader', data_loader.train_dataloader)
