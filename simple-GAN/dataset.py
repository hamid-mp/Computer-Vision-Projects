import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset




class CatsDataset(Dataset):
    
    def __init__(self, root_pth):
        
        self.root = Path(root_pth)
        self.images = list(self.root.iterdir())
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        cls = torch.tensor([1.]) # for descriminator => '1' means real images
        return img, cls
        
        
        
        
