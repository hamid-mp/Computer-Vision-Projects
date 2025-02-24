import torch
import matplotlib.pyplot as plt
import numpy as np




def sample_noise(batch_size, dim, seed=None, device='cpu'): 
    tensor = torch.randn(batch_size, dim, 1, 1, device = device)
    return tensor

    
def noise_gen(n_samples, n_vector_dim, device='cpu'):
    noise = torch.randn(n_samples, n_vector_dim, device=device)  # Generate noise
    noise = noise.view(n_samples, n_vector_dim, 1)  # Reshape to match the input size of the generator
    return noise

class UnNormalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, tensor):
        return tensor.detach().cpu() * self.std + self.mean



def show_images(images, epoch=None, out="./out/", grid_size=(8, 8)):
    # Reshape images to (batch_size, D)
    images = torch.reshape(images, [images.shape[0], -1])
    
    # Initialize UnNormalize
    unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    
    # Define grid size based on desired tiling (e.g., 8x8 grid)
    grid_height, grid_width = grid_size
    assert images.shape[0] >= grid_height * grid_width, "Not enough images for the specified grid size"
    
    # Prepare an empty grid to hold the images
    grid_img = np.zeros((grid_height * 64, grid_width * 64, 3), dtype=np.uint8)
    
    # Iterate through images and place them in the grid
    for i, img in enumerate(images[:grid_height * grid_width]):  # Limit to the number of grid spaces
        img = img.reshape(3, 64, 64)  # Reshape back to (3, 64, 64)
        img = unorm(img)  # Un-normalize the image
        img = img.detach().numpy()  # Convert tensor to numpy array
        img = np.moveaxis(img, 0, -1)  # Move channels to last dimension (H, W, C)
        
        # Normalize the image to the [0, 255] range
        img = ((img - img.min()) * 255) / (img.max() - img.min())
        img = img.astype(np.uint8)  # Convert to uint8
        
        # Determine position in the grid (row and column)
        row = i // grid_width
        col = i % grid_width
        
        # Place the image into the corresponding grid position
        grid_img[row * 64:(row + 1) * 64, col * 64:(col + 1) * 64] = img

    # Save the tiled grid image
    plt.imsave(out + f"{epoch:03d}.jpg", grid_img)
