#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Gil Fernandes
# DATE CREATED: 2018-12-27
# REVISED DATE:
# PURPOSE: Used to train the flowers data set with using multiple architectures including 'vgg16', 'vgg19', 'alexnet', 'densenet161', 'resnet152'
#
#
from train_get_input_args import get_input_args
from train_data_loader import DataLoader
from train_model import VisualModel
from utils.model_persistence import ModelPersistence
from train_model_trainer import ModelTrainer

import os
import sys

args = get_input_args()

print('Application arguments: ', vars(args))

if not os.path.exists(args.data_dir[0]):
    sys.stderr.write(f'Please make sure that the data directory exists and contains the training images. Cannot find {args.data_dir[0]}')
    sys.exit(-1)

data_loader = DataLoader(args.data_dir[0], image_size=299 if args.arch == 'inception_v3' else 224)

print('Data loader ready')
visual_model = VisualModel(args.arch, hidden_layers=args.hidden_units)
print('Visual model ready')
print(visual_model.model)
model_trainer = ModelTrainer(visual_model.model, data_loader.train_dataloader, data_loader.validation_dataloader,
                             args.learning_rate, args.epochs, args.gpu, args.arch, args.print_every)
print('Model trainer ready')
model = model_trainer.train()
model_persistence = ModelPersistence()

check_point_dir = args.save_dir
if not os.path.exists(check_point_dir):
    os.makedirs(check_point_dir)

check_point_file = f"{check_point_dir}/{args.arch}_checkpoint.pth"
print(f"Saving checkpoint in file {check_point_file}")
model_persistence.save_model(data_loader.train_dataset.class_to_idx, visual_model, file=check_point_file)
print(f"Checkpoint saved in file {check_point_file}")
