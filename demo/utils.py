import numpy as np
import torch
import torchvision

def make_grid2d(x, y):
    coords_x = torch.linspace(-1, 1, x)
    coords_y = torch.linspace(-1, 1, y)
    grid = torch.stack(torch.meshgrid(
        coords_x,
        coords_y), -1)

    return grid


def make_grid3d(T, H, W):
    y = torch.linspace(-1, 1, H)
    x = torch.linspace(-1, 1, W)
    t = torch.linspace(-1, 1, T)
    input_grid = torch.stack(torch.meshgrid(t,y,x),-1)
    input_grid = torch.stack((input_grid[:,:,:,0],input_grid[:,:,:,2], input_grid[:,:,:,1]),-1)

    return input_grid

    
def show_tensor_to_image(tensor, file_name):
    torchvision.utils.save_image(tensor, '{}.png'.format(file_name))


def sample_coords(x, sampling_rate, index = 1):
    N_samples = int(sampling_rate * x.shape[index])
    coord_idx = torch.randint(0, x.shape[index], (N_samples, ), device = 'cuda')
    #coord_idx = torch.randperm(x.shape[1], device = 'cuda')[:N_samples]

    return coord_idx