# Import Neccessary Packages
import yaml
import torch
import torch.nn as nn
import imgaug
from dataset import CatsDataset
from torch.utils.data import DataLoader

# Load Variables
with open('./configs.yaml', 'r') as f:
    configs = yaml.load(f, Loader=yaml.SafeLoader)
    

# Read Dataset



cat_dataset = CatsDataset('./cats')
cat_loader = DataLoader(cat_dataset, shuffle=True, batch_size=configs['batch_size'])



for x, y in cat_loader:
    print(y)