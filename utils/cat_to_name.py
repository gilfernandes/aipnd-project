#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Gil Fernandes
# DATE CREATED: 2018-12-27
# REVISED DATE:
# PURPOSE: Used to load the category to name mappings from a JSON file
#
#
import json

class CategoryNameMapping():
    def __init__(self, cat_to_name_file) -> None:
        with open(cat_to_name_file, 'r') as f:
            self.cat_to_name = json.load(f)

