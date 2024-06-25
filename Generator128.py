
import torch.nn as nn
import torch

class Generator(nn.Module):
    
    DROPOUT_PERCENTAGE = 0.25
    CHANNELS = 3
    LATENT_DIM = 100
    
    def __init__(self):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            *self.__create_conv_layer(self.LATENT_DIM, 1024, kernel_size=4, stride=1, padding=0),
            *self.__create_conv_layer(1024, 512),
            *self.__create_conv_layer(512, 256),
            *self.__create_conv_layer(256, 128),
            *self.__create_conv_layer(128, 64),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh(),
        )

        
    def __create_conv_layer(self, in_channels: int, out_channels: int, batch_norm = True, dropout = True,
                 kernel_size=4, stride=2, padding=1) -> list:
        layer = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)]

        layer.append(nn.ReLU(True))
        if batch_norm:
            layer.append(nn.BatchNorm2d(out_channels))

        if dropout:
            layer.append(nn.Dropout2d(self.DROPOUT_PERCENTAGE))
        
        return layer

    def forward(self, x):
        return self.model(x)