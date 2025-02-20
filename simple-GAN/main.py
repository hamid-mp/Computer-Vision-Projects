# Import Neccessary Packages
import torch.utils
import yaml
import torch
import torch.nn as nn
import imgaug.augmenters as iaa
from tqdm import tqdm
from utils import noise_gen, show_images
from models import Generator, Discriminator
from dataset import CatsDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Load Variables
with open('./configs.yaml', 'r') as f:
    configs = yaml.load(f, Loader=yaml.SafeLoader)
    

# config augmentations
img_aug = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Crop(percent=(0, 0.1)),
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    iaa.LinearContrast((0.75, 1.5)),

    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True) 


# Load Dataset
cat_dataset = CatsDataset('./cats', transform=img_aug, device=device)
cat_loader = DataLoader(cat_dataset, shuffle=True, batch_size=configs['batch_size'])



# Load Training Configs
gen_model = Generator().to(device)
dis_model = Discriminator().to(device)
dis_criterion = nn.BCELoss()
gen_criterion = nn.BCELoss()
gen_opt = torch.optim.Adam(gen_model.parameters(), lr=configs['lr'])
dis_opt = torch.optim.Adam(dis_model.parameters(), lr=configs['lr'])


for epoch in range(configs['EPOCHS']):
    gen_model.train()  # Set generator to training mode
    dis_model.train()  # Set discriminator to training mode
    with tqdm(total=len(cat_loader), desc=f'Epoch {epoch+1}/{configs["EPOCHS"]}', ncols=100, unit='batch') as pbar:

        for i, (x_real, y_real) in enumerate(cat_loader):
            x_real, y_real = x_real.to(device), y_real.to(device)
            
            # Generate random noise for the generator's input
            n_sample_noise = noise_gen(configs['batch_size'], 10, device=device)  # noise input for generator
            
            # Generate fake images using the generator
            x_fake = gen_model(n_sample_noise)
            y_fake = torch.zeros([configs['batch_size'], 1], dtype=torch.float32, device=device)  # Fake labels (0)
            
            # Get the discriminator's predictions for real and fake images
            y_pred_fake = dis_model(x_fake)  # Discriminator output for fake images
            y_pred_real = dis_model(x_real)  # Discriminator output for real images
            
            # Calculate the discriminator loss
            loss_real = dis_criterion(y_pred_real, y_real)
            loss_fake = dis_criterion(y_pred_fake, y_fake)
            dis_loss =  loss_fake + loss_real  # Real + Fake loss
            
            # Update discriminator weights
            dis_opt.zero_grad()  # Clear previous gradients
            dis_loss.backward()  
            dis_opt.step()  # Update discriminator parameters
            
            # Calculate generator loss (how well the generator fools the discriminator)
            gen_loss = torch.clone(loss_real).detach().requires_grad_()
            
            # Update generator weights
            gen_opt.zero_grad()  # Clear previous gradients
            gen_loss.backward()  # Backpropagate generator loss
            gen_opt.step()  # Update generator parameters
            
        show_images(x_fake)