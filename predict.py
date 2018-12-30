#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Gil Fernandes
# DATE CREATED: 2018-12-27
# REVISED DATE:
# PURPOSE: Used to make predictions for a photo on the command line using a pre-trained model
#
#
from predict_get_input_args import get_input_args
from predict_service import PredictService
from utils.model_persistence import ModelPersistence

args = get_input_args()
model_persistence = ModelPersistence()
model = model_persistence.load_model(args.gpu, args.checkpoint[0])
predict_service = PredictService(102)
probs, idx = predict_service.predict(args.single_image[0], model, args.gpu, args.top_k)
print(f'Category predictions for {args.single_image}', predict_service.map_to_category(args.category_names, idx))