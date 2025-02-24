# Import Neccessary Packages
import torch.utils
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from utils import noise_gen, show_images, sample_noise
from models import Generator, Discriminator, build_dc_generator, build_dc_classifier
from dataset import CatsDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import v2
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Variables
with open('./configs.yaml', 'r') as f:
    configs = yaml.load(f, Loader=yaml.SafeLoader)
    

# config augmentations
def gauss_noise_tensor(img):
    assert isinstance(img, torch.Tensor)
    dtype = img.dtype
    if not img.is_floating_point():
        img = img.to(torch.float32)
    
    sigma = 25.0
    
    out = img + sigma * torch.randn_like(img)
    
    if out.dtype != dtype:
        out = out.to(dtype)
        
    return out

img_aug = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                             transforms.RandomGrayscale(p=0.2),
                             transforms.RandomVerticalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize(mean = [0.5, 0.5, 0.5],std = [0.5, 0.5, 0.5]  )])





# Load Dataset
cat_dataset = CatsDataset('./cats', transform=img_aug, device=device)
cat_loader = DataLoader(cat_dataset, shuffle=True, batch_size=configs['batch_size'])



# Load Training Configs
gen_model = build_dc_generator(noise_dim=configs['noise_dim']).to(device)#(noise_dim=configs['noise_dim']).to(device)
gen_model = nn.DataParallel(gen_model)
dis_model =  build_dc_classifier().to(device)#Discriminator().to(device)
dis_model = nn.DataParallel(dis_model)
dis_criterion = nn.BCELoss()
gen_criterion = nn.BCELoss()

gen_opt = torch.optim.Adam(gen_model.parameters(), lr=configs['gen_lr'])
dis_opt = torch.optim.Adam(dis_model.parameters(), lr=configs['dis_lr'], betas=(0.5, 0.999))

gen_loss_epochs, dis_loss_epochs = [], []
for epoch in range(configs['EPOCHS']):
    
    
    dis_epoch_loss, gen_epoch_loss = 0, 0
    

    gen_model.train()  # Set generator to training mode
    dis_model.train()  # Set discriminator to training mode
    with tqdm(total=len(cat_loader), desc=f'Epoch {epoch+1}/{configs["EPOCHS"]}', ncols=100, unit='batch') as pbar:
        for i, (x_real, y_real) in enumerate(cat_loader):
             
            x_real, y_real = x_real.to(device), y_real.to(device)
            
            # Generate random noise for the generator's input
            n_sample_noise = sample_noise(configs['batch_size'], dim=configs['noise_dim'], device=device) 
            # Generate fake images using the generator
            x_fake = gen_model(n_sample_noise)
            y_fake = torch.zeros([configs['batch_size'], 1], dtype=torch.float32, device=device)  # Fake labels (0)


            # Get the discriminator's predictions for real and fake images
            y_pred_fake = dis_model(x_fake)  # Discriminator output for fake images
            y_pred_real = dis_model(x_real)  # Discriminator output for real images
            dis_opt.zero_grad()  # Clear previous gradients
            # Calculate the discriminator loss
            loss_real = dis_criterion(y_pred_real.view(-1, 1), y_real)
            loss_fake = dis_criterion(y_pred_fake.view(-1, 1), y_fake)
            dis_loss =  (loss_fake + loss_real)   # Real + Fake loss
            dis_epoch_loss += dis_loss.item()
            # Update discriminator weights
            dis_loss.backward()  
            dis_opt.step()  # Update discriminator parameters
            # Calculate generator loss (how well the generator fools the discriminator)

            gen_opt.zero_grad()  # Clear previous gradients
            x_fake = gen_model(n_sample_noise)
            new_y_fake_out = dis_model(x_fake)  # Discriminator output for fake images
            y_real = torch.ones_like(new_y_fake_out)
            gen_loss = gen_criterion(new_y_fake_out, y_real)
            gen_epoch_loss += gen_loss.item()
            # Update generator weights
            
            gen_loss.backward()  # Backpropagate generator loss
            gen_opt.step()  # Update generator parameters
            pbar.update(1)
        dis_epoch_loss /= len(cat_loader)    
        gen_epoch_loss /= len(cat_loader)    
        gen_loss_epochs.append(gen_epoch_loss)
        dis_loss_epochs.append(dis_epoch_loss)
        

        fig, ax = plt.subplots(figsize=(8, 6))
        # plt.ylim((0, 1e-4))

        plt.plot(list(range(epoch+1)), gen_loss_epochs, label='G loss')
        plt.title('Genrator training loss')
        plt.savefig("Generator loss.png", dpi=300)
        plt.cla()
        plt.close()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.plot(list(range(epoch+1)), dis_loss_epochs, label='D loss')
        plt.title('Discriminator training loss')
        # plt.ylim((0, 1e-4))
        plt.savefig("Discriminator loss.png", dpi=300)
        plt.cla()
        plt.close()
        
        show_images(x_fake, epoch)
        
    if epoch % 5 == 0:
        Path('./weights/').mkdir(exist_ok=True, parents=True)
        torch.save(gen_model.state_dict(), './weights/generator.pt')
