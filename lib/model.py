import torch
import torch.nn as nn
import numpy as np

class AffineCouplingLayer(nn.Module):
    def __init__(self, mask, width):
        '''
        Inspired heavilly by github.com/xqding/RealNVP

        Assume this layer maps data of size [b,n] -> [b,n] where b is a batch dimension and n is the size of a datum

        mask - an array of 0s and 1s of size (n)
               this layer will update the values with mask = 0
        '''
        super().__init__()

        self.mask = mask
        self.n = len(mask)

        self.width = width #nodes in hidden layer

        #Layers for scale
        self.scale_layer1 = nn.Linear( self.n,     self.width )
        self.scale_layer2 = nn.Linear( self.width, self.width )
        self.scale_layer3 = nn.Linear( self.width, self.n     )

        #Layers for translation
        self.trans_layer1 = nn.Linear( self.n,     self.width )
        self.trans_layer2 = nn.Linear( self.width, self.width )
        self.trans_layer3 = nn.Linear( self.width, self.n     )

    def scale(self, x):
        x = x*self.mask #mask out values we cannot use in update
        activation = nn.Tanh()
        x = activation(self.scale_layer1(x))
        x = activation(self.scale_layer2(x))
        x = self.scale_layer3(x) #no activation on last layer
        return x

    def translation(self, x):
        x = x*self.mask #mask out values we cannot use in update
        activation = nn.Tanh()
        x = activation(self.trans_layer1(x))
        x = activation(self.trans_layer2(x))
        x = self.trans_layer3(x) #no activation on last layer
        return x

    def forward(self, z):
        # map latent varible z to observed variable x
        s = self.scale(z)
        t = self.translation(z)
        
        x = self.mask*z + ( 1 - self.mask )*( z * torch.exp(s) + t)        
        logdet = torch.sum((1 - self.mask)*s, dim=1)
        
        return x, logdet
    
    def inverse(self, x):
        # map observed varible x to latent space variable z
        s = self.scale(x)
        t = self.translation(x)
                
        z = self.mask*x + (1-self.mask)*((x - t)*torch.exp(-s))
        logdet = torch.sum((1 - self.mask)*(-s), dim=1)
        
        return z, logdet


class SimpleNN(nn.Module):
    def __init__(self, num_affine_layers, width, mask1, mask2 ):
        super().__init__()

        module_list = nn.ModuleList() #Made a list of layers to train
        
        #Add several layers alternating the modified coordinate
        for _ in np.arange( num_affine_layers ):
            module_list.append( AffineCouplingLayer(mask1, width) )
            module_list.append( AffineCouplingLayer(mask2, width) )
        
        self.module_list = module_list
    
    def forward(self, z):
        #Map the Gaussian variable z to x
        log_det = 0.0
        for layer in self.module_list:
            z, mini_log_det = layer.forward( z )
            log_det = log_det + mini_log_det
        return z, log_det
    
    def inverse(self, x):
        #Map the x to Gaussian z
        log_det = 0.0
        n = len(self.module_list)

        #apply them in reverse order!
        for i in np.arange( n ):
            layer = self.module_list[n-i-1]
            x, mini_log_det = layer.inverse( x )
            log_det = log_det + mini_log_det
        return x, log_det