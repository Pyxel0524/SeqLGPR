""" Full assembly of the parts to form the complete network """
import torch.nn.functional as F
import torch.nn as nn

from .unet.unet_model import UNet


class SiamSO(nn.Module):
    def __init__(self, n_channels,  classes, bilinear=False):
        super(SiamSO, self).__init__()
        self.bilinear = bilinear
        self.classes = classes
        self.unet = UNet(n_channels,classes, bilinear)
        self.match_batchnorm = nn.BatchNorm2d(1)

    def forward(self, temp, search):
        b, c, h, w = search.shape
        template_feature = self.unet(temp)
        search_feature = self.unet(search)
        # print(template_feature.shape)
        # # print(search_feature.shape)
        match_map = F.conv2d(search_feature.view(1, b*c, search_feature.shape[-2], search_feature.shape[-1]),
                          weight = template_feature.view(-1, 1, template_feature.shape[-2], template_feature.shape[-1]),
                          groups = b * self.classes)
        match_map = F.interpolate(match_map, size = (temp.shape[-2],temp.shape[-1]))
        
        # Here we reorder the dimensions to get back the batch dimension.
        match_map = match_map.permute(1, 0, 2, 3)
        match_map = self.match_batchnorm(match_map)

        return match_map 
