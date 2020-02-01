""" Full assembly of the parts to form the complete network """
import torch
import torch.nn.functional as F

from torch import nn

from unet_parts import *


class Unet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256) #self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)
        #self.outm = OutConv(192, n_classes)

    def forward(self, x0):
        x1 = self.inc(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        '''
        y1 = self.inc(x0)
        y2 = self.down1(y1)
        y3 = self.down2(y2)
        y4 = self.down3(y3)
        y5 = self.down4(y4)
        y = self.up1(y5, y4)
        y = self.up2(y, y3)
        y = self.up3(y, y2)
        y = self.up4(y, y1)
        
        z1 = self.inc(x0)
        z2 = self.down1(z1)
        z3 = self.down2(z2)
        z4 = self.down3(z3)
        z5 = self.down4(z4)
        z = self.up1(z5, z4)
        z = self.up2(z, z3)
        z = self.up3(z, z2)
        z = self.up4(z, z1)
         '''
        #out = torch.cat([x,y,z],dim=1)
        #logits = self.outm(out)
        logits = self.outc(x)
        return logits
