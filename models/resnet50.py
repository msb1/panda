import torch
from torch import nn


class ResNet50(nn.Module):
    def __init__(self, batch_size=8, num_chan=3, classes=6, dropout=0.1):
        super(ResNet50, self).__init__()

        self.batch_size = batch_size
        self.num_chan = num_chan
        self.classes = classes
        self.dropout = dropout
        self.batch_size = batch_size

        # Conv1 layers (start default dim --> 224)
        self.conv1 = nn.Conv2d(num_chan, 64, kernel_size=7, stride=2, padding=3, bias=True)      # redux 2 --> 112

        # Conv2 layers
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)                         # redux 2 --> 56
        self.conv2A = self.convBlock(64, 64, 256)
        self.conv2B = self.convBlock(256, 64, 256)
        self.bypass2 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=True)

        # Conv3 layers
        self.conv3A = self.convBlock(256, 128, 512, kernel_size=3, padding=1, stride=2)              # redux 2 --> 28
        self.conv3B = self.convBlock(512, 128, 512)
        self.bypass3 = nn.Conv2d(256, 512, kernel_size=1, stride=2, padding=0, bias=True)

        # Conv4 layers
        self.conv4A = self.convBlock(512, 256, 1024, kernel_size=3, padding=1, stride=2)             # redux 2 --> 14
        self.conv4B = self.convBlock(1024, 256, 1024)
        self.bypass4 = nn.Conv2d(512, 1024, kernel_size=1, stride=2, padding=0, bias=True)

        # Conv5 layers
        self.conv5A = self.convBlock(1024, 512, 2048, kernel_size=3, padding=1, stride=2)             # redux 2 --> 7
        self.conv5B = self.convBlock(2048, 512, 2048)
        self.bypass5 = nn.Conv2d(1024, 2048, kernel_size=1, stride=2, padding=0, bias=True)

        # Misc layers
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d(1)              # global average pooling collapse each 7x7 plane to 1 --> (batch_size, 2048, 1, 1)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)  # flatten --> (batch_size, 2048)
        self.dense = nn.Linear(2048, classes)


    def convBlock(self, dim1, dim2, dim3, kernel_size=3, padding=1, stride=1):
        return nn.Sequential(
            nn.Conv2d(dim1, dim2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(dim2),
            nn.ReLU(),

            nn.Conv2d(dim2, dim2, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(dim2),
            nn.ReLU(),

            nn.Conv2d(dim2, dim3, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(dim3),
        )


    def forward(self, x0):
        # input layer
        # print("input", x0.shape)
        x = self.conv1(x0)
        # print("after conv1", x.shape)
        # layer 2
        x = self.maxpool(x)
        # print("after max pool", x.shape)
        xbypass = x
        x = self.conv2A(x)
        x = self.conv2B(x)
        x = self.conv2B(x) + self.bypass2(xbypass)
        # layer 3
        xbypass = x
        # print("start layer 3", x.shape)
        x = self.conv3A(x)
        x = self.conv3B(x)
        x = self.conv3B(x)
        x = self.conv3B(x) + self.bypass3(xbypass)
        # layer 4
        xbypass = x
        x = self.conv4A(x)
        x = self.conv4B(x)
        x = self.conv4B(x)
        x = self.conv4B(x)
        x = self.conv4B(x)
        x = self.conv4B(x) + self.bypass4(xbypass)
        # layer 5
        xbypass = x
        x = self.conv5A(x)
        x = self.conv5B(x)
        x = self.conv5B(x) + self.bypass5(xbypass)
        # output layer
        x = self.relu(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x
