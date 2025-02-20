import torch
import torch.nn as nn



class Generator(nn.Module):
    '''
    Generates random images, output must be the size of an image (batch_size, 3, 64, 64)
    '''
    def __init__(self, noise_dim=10, image_dim=3, hidden_dim=128):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        self.generator = nn.Sequential(
            self.generator_block(self.noise_dim, self.hidden_dim * 8),
            self.generator_block(self.hidden_dim * 8, self.hidden_dim * 4),
            self.generator_block(self.hidden_dim * 4, self.hidden_dim * 2),
            self.generator_block(self.hidden_dim * 2, self.hidden_dim*4),
            nn.ConvTranspose2d(self.hidden_dim*4, self.hidden_dim*8, kernel_size=4, stride=2, padding=1),  # Final layer to get (64x64x3)
            nn.ConvTranspose2d(self.hidden_dim*8, self.image_dim, kernel_size=4, stride=2, padding=1),  # Final layer to get (64x64x3)

            nn.Tanh()  # Use Tanh activation to scale the output between -1 and 1
        )
        
    def generator_block(self, input_ch, output_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(input_ch, output_ch, kernel_size=4, stride=2, padding=1),  # Upsampling (conv transpose)
            nn.BatchNorm2d(output_ch),  # Batch normalization
            nn.ReLU(inplace=True)  # ReLU activation
        )
    
    def forward(self, input_noise):
        x = input_noise.view(input_noise.size(0), self.noise_dim, 1, 1)  # Reshape noise to [batch_size, noise_dim, 1, 1]
        return self.generator(x)




class Discriminator(nn.Module):
    def __init__(self, img_ch=3, hidden_ch=8):
        super(Discriminator, self).__init__()
        self.i_ch = img_ch
        self.h_ch = hidden_ch
        self.discriminator = nn.Sequential(
            self.discriminator_block(self.i_ch, self.h_ch),
            self.discriminator_block(self.h_ch, self.h_ch*2),
            self.discriminator_block(self.h_ch*2, self.h_ch*4),
            nn.Flatten(),                                          
            nn.Linear(self.h_ch * 4 * 64 * 64, 1), 
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.discriminator(x)
    
    
    def discriminator_block(self, input_channles, output_channels):
        return nn.Sequential(nn.Conv2d(input_channles, output_channels, kernel_size=3, padding=1, stride=1), # the shape is constant => n-k+2p / s
                             nn.BatchNorm2d(output_channels),
                             nn.ReLU(inplace=True))