#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_input_args.py
#
# PROGRAMMER: Gil Fernandes
# DATE CREATED: 2018-12-27
# REVISED DATE:
# PURPOSE: Parse the command line arguments in order to train the neural network
#
#
import argparse


def get_input_args():
    """
    Options:
    Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
    Choose architecture: python train.py data_dir --arch "vgg13"
    Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
    Use GPU for training: python train.py data_dir --gpu
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(
        description='Process Image Folder, CNN Model Architecture, Set hyper parameters')
    # Create 3 command line arguments as mentioned above using add_argument() from ArgumentParser method
    parser.add_argument('data_dir', metavar='data_dir', type=str, nargs=1,
                        help='The directory of the image data')
    parser.add_argument('--arch', type=str, default='vgg16',
                        help='CNN Model Architecture with default value \'vgg16\', \'vgg19\', \'alexnet\', \'densenet161\'')
    parser.add_argument('--save_dir', type=str, default='/tmp',
                        help='The directory to which the chepoints are saved with default value \'/tmp\'')
    parser.add_argument('--learning_rate', type=float, default='0.001',
                        help='The learning rate floating point number with default value \'0.001\'')
    parser.add_argument('--hidden_units', type=int, default='512',
                        help='The number of units in the hidden layer with default value \'512\'')
    parser.add_argument('--epochs', type=int, default='20',
                        help='The number of epochs \'20\'')
    parser.add_argument('--gpu', action='store_true',
                        help='If available then the GPU will be used, else not')
    parser.add_argument('--catnamefile', type=str, default='cat_to_name.json',
                        help='The file containing the mappings of the category id to the actual name with default value \'cat_to_name.json\'')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_input_args()
    print('data_dir', args.data_dir[0])
    print('hidden_units', args.hidden_units)
    print('epochs', args.epochs)
    print('gpu', args.gpu)
    print('catnamefile', args.catnamefile)
