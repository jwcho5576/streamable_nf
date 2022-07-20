import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class Kodak():
    def __init__(self, args, num, device):
        self.args = args
        self.device = device
        self.y = Image.open(f'./data/kodak/kodim{num:02}.png')
        self.y = transforms.ToTensor()(self.y).to(device)
        self.C, self.H, self.W = self.y.shape
        self.y = self.y.reshape(self.C, -1).permute(1, 0)
        self.x = self.make_grid(self.H, self.W).to(device)
    
    def mask(self, input, w_idx, w_num, eval=False):
        # spatial masking for spatial growing
        out = input.clone()
        if input.dim() == 2:
            out = out.permute(1, 0).reshape(self.C, self.H, self.W)
        window_width = int(self.W/w_num)
        if eval:
            # remove masked part when evaluating metrics
            indices = torch.arange(0, window_width*(w_idx+1))
            out = out[:, :, indices]
            return out
        else:
            indices = torch.arange(window_width*(w_idx+1), self.W, device=self.device)
            if self.args.model == 'individual':
                indices = torch.cat((indices, torch.arange(0, window_width*w_idx, device=self.device)))
            out.index_fill_(2, indices, 0)
            return out.reshape(self.C, -1).permute(1, 0)

    def make_grid(self, H, W):
        coords_x = torch.linspace(-1, 1, H)
        coords_y = torch.linspace(-1, 1, W)
        grid = torch.stack(torch.meshgrid(
            coords_x,
            coords_y,
            indexing='ij'), -1)

        return grid.reshape(-1, 2)


class UVG():
    def __init__(self, args, name, device):
        self.args = args
        self.device = device

        # create frames

        frames = sorted(glob.glob(os.path.join(f'./data/uvg/{name}', f'{name}_*.jpg')))

        self.n_frames = len(frames)
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((270, 480))
        ])

        frames = [
            tf(Image.open(f))
            for f in frames[:self.n_frames]
        ]
        self.y = torch.stack(frames, 0).to(device)
        self.C, self.H, self.W = self.y.shape[1:]
        self.y = self.y.reshape(self.n_frames, self.C, -1).permute(0, 2, 1)
        self.x = self.make_grid(self.n_frames, self.H, self.W).to(device)

    def mask(self, input, w_idx, w_num, eval=False):
        # temporal masking for temporal growing
        out = input.clone()
        n_frames = out.shape[0]
        frame_window = int(n_frames/w_num)
        if eval:
            indices = torch.arange(0, frame_window*(w_idx+1))
            out = out[indices, :, :]
            return out
        else:
            indices = torch.arange(frame_window*(w_idx+1), n_frames, device=self.device)
            if self.args.model == 'individual':
                indices = torch.cat((indices, torch.arange(0, frame_window*w_idx, device=self.device)))
            out.index_fill_(0, indices, 0)
            return out

    def make_grid(self, T, H, W):
        coords_t = torch.linspace(-1, 1, T)
        coords_x = torch.linspace(-1, 1, H)
        coords_y = torch.linspace(-1, 1, W)
        
        grid = torch.stack(torch.meshgrid(
            coords_t,
            coords_x,
            coords_y,
            indexing='ij'), -1)

        return grid.reshape(self.n_frames, -1, 3)


class PointCloud(Dataset):
    '''
    modified from https://github.com/vsitzmann/siren/blob/master/dataio.py
    '''
    def __init__(self, pointcloud_path, on_surface_points, keep_aspect_ratio=True):
        super().__init__()

        # just dummy tensors for building networks in main()
        self.x = torch.empty((3))
        self.y = torch.empty((1))

        print("Loading point cloud")
        point_cloud = np.genfromtxt(pointcloud_path)
        print("point_cloud", point_cloud.shape)
        print("Finished loading point cloud")
        self.n_points = point_cloud.shape[0]

        coords = point_cloud[:, :3]
        #pcd = o3d.io.read_point_cloud(pointcloud_path)
        #o3d.geometry.PointCloud.estimate_normals(pcd, search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        #self.normals = np.asarray(pcd.normals)
        self.normals = point_cloud[:, 3:]


        # Reshape point cloud such that it lies in bounding box of (-1, 1) (distorts geometry, but makes for high
        # sample efficiency)
        coords -= np.mean(coords, axis=0, keepdims=True)
        if keep_aspect_ratio:
            coord_max = np.amax(coords)
            coord_min = np.amin(coords)
        else:
            coord_max = np.amax(coords, axis=0, keepdims=True)
            coord_min = np.amin(coords, axis=0, keepdims=True)
        
        #rescale
        self.coords = (coords - coord_min) / (coord_max - coord_min)
        self.coords -= 0.5
        self.coords *= 2.

        self.on_surface_points = on_surface_points

    def __len__(self):
        return self.coords.shape[0] // self.on_surface_points

    def __getitem__(self, idx):
        point_cloud_size = self.coords.shape[0]

        off_surface_samples = self.on_surface_points  # **2
        total_samples = self.on_surface_points + off_surface_samples

        # Random coords
        rand_idcs = np.random.choice(point_cloud_size, size=self.on_surface_points)

        on_surface_coords = self.coords[rand_idcs, :]
        on_surface_normals = self.normals[rand_idcs, :]

        off_surface_coords = np.random.uniform(-1, 1, size=(off_surface_samples, 3))
        off_surface_normals = np.ones((off_surface_samples, 3)) * -1

        sdf = np.zeros((total_samples, 1))  # on-surface = 0
        sdf[self.on_surface_points:, :] = -1  # off-surface = -1

        coords = np.concatenate((on_surface_coords, off_surface_coords), axis=0)
        normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)

        return {'coords': torch.from_numpy(coords).float()}, {'sdf': torch.from_numpy(sdf).float(),
                                                              'normals': torch.from_numpy(normals).float()}