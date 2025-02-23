import torch
from pathlib import Path
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms




class CatsDataset(Dataset):
    
    def __init__(self, root_pth, transform=None, device='cpu'):
        self.device = device
        self.root = Path(root_pth)
        self.images = list(self.root.iterdir())


        self.transform = transform if transform else transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        cls = torch.tensor([1.], dtype=torch.float32, device=self.device) # for descriminator => '1' means real images
        return img, cls
        
        
        
        
