'''
Author: Anthony Badea
Date: March 18, 2023
'''

import torch
import torch.nn as nn
import numpy as np

#%%%%%%% Classes %%%%%%%%#

class DNN_block(nn.Module):

    def __init__(self, dimensions, normalize_input):

        super().__init__()
        input_dim = dimensions[0]
        self.input_bn = nn.BatchNorm1d(input_dim) if normalize_input else None
        self.input_ln = nn.LayerNorm(input_dim) if normalize_input else None

        layers = []
        for iD, (dim_in, dim_out) in enumerate(zip(dimensions, dimensions[1:])):
            if iD != len(dimensions[1:])-1:
                layers.extend([
                    nn.Linear(dim_in, dim_out),
                    nn.LayerNorm(dim_out),
                    nn.ReLU(),
                ])
            else:
                layers.extend([
                    nn.Linear(dim_in, dim_out),
                ])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if self.input_bn is not None:
            if len(x.shape) ==2:
            	#x = self.input_bn(x)
            	x = self.input_ln(x)
            elif len(x.shape) >=3:
            	x = self.input_bn(x.transpose(1,2)) # must transpose because batchnorm expects N,C,L rather than N,L,C like multihead
            	x = x.transpose(1,2) # revert transpose
        return self.net(x)


class SignatureDiscovery(nn.Module):

    def __init__(
        self, 
        embed_dimensions,
        embed_normalize_input,
        bkg_dimensions,
        bkg_normalize_input
    ):

        super().__init__()

        # model
        self.embed = DNN_block(embed_dimensions, embed_normalize_input)
        self.bkg = DNN_block(bkg_dimensions, bkg_normalize_input)

    def forward(self, x):

        e = self.embed(x)
        b = self.bkg(e)
        return e, b

if __name__ == "__main__":

    x = torch.Tensor(15,10) # batch x features
    m = SignatureDiscovery([10, 10, 5], False, [5, 3, 1], False)
    e, b = m(x)
    print(e.shape, b.shape)
        