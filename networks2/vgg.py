from torch import nn
import torch
import torch.nn.functional as F
# from modules.util import AntiAliasInterpolation2d, make_coordinate_grid
from torchvision import models
import numpy as np
from torch.autograd import grad


class Vgg16_bn(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """
    def __init__(self, requires_grad=True):
        super(Vgg16_bn, self).__init__()
        vgg_pretrained_features = models.vgg16_bn(pretrained=True).features
        # print(vgg_pretrained_features)
        # weights_path = "path_to_pretrained_vgg16-397923af.pth"
        # vgg_pretrained_features.load_state_dict(torch.load(weights_path))
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(6):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 13):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(14, 24):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(24, 37):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

    def forward(self, X):
        # X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        out = [h_relu2, h_relu3, h_relu4]#may add h_relu1 later
        return out

      
if __name__ == "__main__":
    print_summary = True

    
    bsz=4
    gen = Vgg16_bn()
    x = torch.randn(1,3,56,56)
    y = gen(x)
    print(y[0].size())
    print(y[1].size())
    print(y[2].size())
#     print(y[3].size())
    
