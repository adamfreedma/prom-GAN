from GaussianNoise import GaussianNoise

import torch.nn as nn
import torch


class Discriminator(nn.Module):

    LEAKY_RELU_SLOPE = 0.2
    DROPOUT_PERCENTAGE = 0.25
    CHANNELS = 3
    STARTING_SIZE = 64

    def __init__(self):
        super(Discriminator, self).__init__()

        self.convolutional_model = nn.Sequential(
            *self.__create_conv_layer(3, 64),
            *self.__create_conv_layer(64, 128),
            *self.__create_conv_layer(128, 256),
            *self.__create_conv_layer(256, 512),
        )

        self.flatten_size = self.__get_flatten_size()

        self.fully_connected_model = nn.Sequential(
            *self.__create_fully_connected_layer(self.flatten_size, 512),
            *self.__create_fully_connected_layer(512, 1),
            nn.Sigmoid(),
        )

    def __get_flatten_size(self):
        dummy_input = torch.zeros(1, 3, self.STARTING_SIZE, self.STARTING_SIZE)
        # Pass the dummy input through the convolutional layers
        dummy_result = self.convolutional_model(dummy_input)
        # Flatten the output to calculate the size
        flatten_size = dummy_result.view(-1).size(0)

        return flatten_size

    def __create_conv_layer(
        self,
        in_channels: int,
        out_channels: int,
        batch_norm=True,
        dropout=True,
        kernel_size=3,
        stride=2,
        padding=1,
    ) -> list:
        layer = [
            GaussianNoise(),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        ]

        layer.append(nn.LeakyReLU(self.LEAKY_RELU_SLOPE, inplace=True))
        if batch_norm:
            layer.append(nn.BatchNorm2d(out_channels))  # TODO: check momentum

        if dropout:
            layer.append(nn.Dropout2d(self.DROPOUT_PERCENTAGE))

        return layer

    def __create_fully_connected_layer(
        self, in_features: int, out_features: int
    ) -> list:
        layer = [
            nn.Linear(in_features, out_features),
            nn.LeakyReLU(self.LEAKY_RELU_SLOPE, inplace=True),
        ]

        return layer

    def forward(self, x):
        x = self.convolutional_model(x)
        x = x.view(-1, self.flatten_size)
        x = self.fully_connected_model(x)

        return x
