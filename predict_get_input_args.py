#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_input_args.py
#
# PROGRAMMER: Gil Fernandes
# DATE CREATED: 2018-12-27
# REVISED DATE:
# PURPOSE: Parse the command line arguments
#
#
import argparse


def get_input_args():
    """
    Used to parse the command line arguments in order to predict the flower name and the class probability.
    Options:
    Return top KK most likely classes: python predict.py input checkpoint --top_k 3
    Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
    Use GPU for inference: python predict.py input checkpoint --gpu
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(
        description='Process Image Folder, CNN Model Architecture, Set hyper parameters')
    parser.add_argument('single_image', metavar='single_image', type=str, nargs=1,
                        help='a single image for which the flower name and the class probability is to be predicted')
    parser.add_argument('checkpoint', metavar='checkpoint', type=str, nargs=1,
                        help='The checkpoint from which the model is re-built for the prediction')
    parser.add_argument('--top_k', type=int, default='3',
                        help='The number of most likely classes with default value \'3\'')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='A file mapping of categories to real names with default value \'cat_to_name.json\'')
    parser.add_argument('--gpu', action='store_true',
                        help='If available then the GPU will be used, else not')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_input_args()
    print("single_image", args.single_image)
    print("checkpoint", args.checkpoint)
    print("--top_k", args.top_k)
    print("--category_names", args.category_names)
    print("--gpu", args.gpu)
