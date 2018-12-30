#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Gil Fernandes
# DATE CREATED: 2018-12-27
# REVISED DATE:
# PURPOSE: Used to train the neural networks and report the progress on the console
#
#
import torch

from torch import nn, optim


class ModelTrainer():
    def __init__(self, model, train_dataloader, validation_dataloader, learning_rate=0.001, epochs=20, gpu=True,
                 arch='vgg16'):
        self.model = model
        self.epochs = epochs
        self.gpu = gpu
        self.criterion = nn.NLLLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,
                                         weight_decay=1e-4) if arch.startswith('resnet') else \
            optim.Adam(model.classifier.parameters(), lr=learning_rate)
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

    def validation(self):
        test_loss = 0
        accuracy = 0
        for images, labels in self.validation_dataloader:
            images, labels = self.convert_to_platform(images, labels)
            output = self.model.forward(images)
            test_loss += self.criterion(output, labels).item()

            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()

        return test_loss, accuracy

    def train(self):

        if self.gpu:
            self.model.to('cuda')
        else:
            self.model.to('cpu')

        steps = 0
        print_every = 40
        for e in range(self.epochs):
            running_loss = 0
            for ii, (inputs, labels) in enumerate(self.train_dataloader):
                steps += 1

                inputs, labels = self.convert_to_platform(inputs, labels)

                self.optimizer.zero_grad()
                # Forward and backward passes
                outputs = self.model.forward(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    # Network is in eval mode for inference
                    self.model.eval()

                    # Turn off gradients for validation, saves memory and computations
                    with torch.no_grad():
                        test_loss, accuracy = self.validation()

                    validation_loader_length = len(self.validation_dataloader)
                    print(f"Epoch: {e + 1}/{self.epochs}... ",
                          f"Training Loss: {running_loss / print_every:.4f} ",
                          f"Test Loss: {test_loss / validation_loader_length :.3f} ",
                          f"Test Accuracy: {accuracy / validation_loader_length:.3f}")

                    running_loss = 0

                    # Training is back on
                    self.model.train()

        return self.model

    def convert_to_platform(self, inputs, labels):
        if self.gpu:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
        return inputs, labels
