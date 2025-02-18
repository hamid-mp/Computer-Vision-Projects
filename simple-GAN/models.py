import torch
import torch.nn as nn



class Generator(nn.Module):
    '''
    Generates random images, output must be the size of an image
    '''
    def __init__(self, noise_dim=10, image_dim=784, hidden_dim=128):
        self.noise_dim = noise_dim
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        self.generator = nn.Sequential([
            self.generator_block(self.noise_dim, self.hidden_dim),
            self.generator_block(self.hidden_dim*2, self.hidden_dim*4),
            self.generator_block(self.hidden_dim*2, self.hidden_dim*4),
            self.generator_block(self.hidden_dim*4, self.hidden_dim*8),
            self.generator_block(self.hidden_dim*8, self.image_dim),
            nn.Sigmoid()

        ])
        
    
    def generator_block(self, input_dim, output_dim):
        
        return nn.Sequential(
            [nn.Linear(input_dim, output_dim),
             nn.BatchNorm1d(output_dim),
             nn.ReLU(inplace=True)])
     
    # input to a generator model is a noise   
    def forward(self, input_noise):
        return self.generator(input_noise)
    

class Discriminator(nn.Module):
    def __init__(self, img_ch, hidden_ch):
        self.i_ch = img_ch
        self.h_ch = hidden_ch
        self.descriminator = nn.Sequential(
            self.descriminator_block(self.i_ch, self.h_ch),
            self.descriminator_block(self.h_ch, self.h_ch*2),
            self.descriminator_block(self.h_ch*2, self.h_ch*4),
            nn.Linear(self.h_ch*4)
        )
    
    
    
    
    def descriminator_block(self, input_channles, output_channels):
        return nn.Sequential(nn.Conv2d(input_channles, output_channels, kernel_size=3, padding=1, stride=1), # the shape is constant => n-k+2p / s
                             nn.BatchNorm2d(output_channels),
                             nn.ReLU(inplace=True))