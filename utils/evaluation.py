import csv
import os

import lpips
import torchvision
import trimesh
from dataset import PointCloud
from PIL import Image
from torch.utils.data import DataLoader

from utils.metrics import *
from utils.utils import *


class ImageEval(object):
    def __init__(self, result_dir, dataset, args):
        self.result_dir = result_dir
        self.dataset = dataset
        self.args = args

    def __call__(self, net, w_idx):
        # show residual output
        if self.args.model == 'progressive' and w_idx > 0:
            yhat_res = net.forward_residual(self.dataset.x).permute(1, 0).reshape(-1, self.dataset.H, self.dataset.W)
            torchvision.utils.save_image(yhat_res, os.path.join(self.result_dir, f'recon_res_{w_idx}.png'))
        
        yhat = net(self.dataset.x).permute(1, 0).reshape(-1, self.dataset.H, self.dataset.W)
        torchvision.utils.save_image(yhat, os.path.join(self.result_dir, f'recon_{w_idx}.png'))

    def metrics(self, n):
        ref = self.dataset.y.permute(1, 0).reshape(-1, self.dataset.H, self.dataset.W).detach().cpu()

        lpips_fn = lpips.LPIPS(net='alex')
        psnrs = []
        ssims = []
        lpipss = []
        print("calculating metrics...")
        for w_idx in range(n):
            yhat = Image.open(os.path.join(self.result_dir, f'recon_{w_idx}.png'))
            yhat = torchvision.transforms.ToTensor()(yhat)

            # used for spatial growing
            if self.args.experiment == 'image_spatial':
                ref = self.dataset.mask(input=self.dataset.y.detach().cpu(), w_idx=w_idx, w_num=n, eval=True)
                # concat outputs of individual model
                if self.args.model == 'individual' and w_idx > 0:
                    yhat = concat_patches(yhat_prev, yhat, w_idx, n)
                yhat = self.dataset.mask(input=yhat, w_idx=w_idx, w_num=n, eval=True)
                yhat_prev = yhat.clone()

            # PSNR
            psnrs.append(round(PSNR()(yhat, ref).item(), 3))
            # SSIM
            ssims.append(round(SSIM()(torch.unsqueeze(yhat, 0), torch.unsqueeze(ref, 0)).item(), 4))
            # LPIPS
            lpipss.append(round(lpips_fn(torch.unsqueeze(yhat, 0), torch.unsqueeze(ref, 0)).item(), 4))

        with open(os.path.join(self.result_dir, 'PSNR_SSIM_LPIPS.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(psnrs)
            writer.writerow(ssims)
            writer.writerow(lpipss)    

        print(f'PSNR: {psnrs}')
        print(f'SSIM: {ssims}')
        print(f'LPIPS: {lpipss}')


class VideoEval(object):
    def __init__(self, result_dir, dataset, args):
        self.result_dir = result_dir
        self.dataset = dataset
        self.args = args

    def __call__(self, net, w_idx):
        for f in range(self.dataset.n_frames):
            yhat = net(self.dataset.x[f]).permute(1, 0).reshape(-1, self.dataset.H, self.dataset.W)
            os.makedirs(os.path.join(self.result_dir, f'frames_subnet_{w_idx}'), exist_ok=True)
            torchvision.utils.save_image(yhat, os.path.join(self.result_dir, f'frames_subnet_{w_idx}/recon_{f}.png'))

            # show residual output
            if self.args.model == 'progressive' and w_idx > 0:
                yhat_res = net.forward_residual(self.dataset.x[f]).permute(1, 0).reshape(-1, self.dataset.H, self.dataset.W)
                torchvision.utils.save_image(yhat_res, os.path.join(self.result_dir, f'frames_subnet_{w_idx}/recon_res_{f}.png'))

    def metrics(self, n):
        lpips_fn = lpips.LPIPS(net='alex')
        psnrs = []
        ssims = []
        lpipss = []
        frame_window = int(self.dataset.n_frames / n)
        print("calculating metrics...")
        for w_idx in range(n):
            ref_full = self.dataset.mask(input=self.dataset.y.detach().cpu(), w_idx=w_idx, w_num=n, eval=True)

            psnr_total = 0
            ssim_total = 0
            lpips_total = 0
            for f in range(ref_full.shape[0]):
                ref = ref_full[f].permute(1, 0).reshape(-1, self.dataset.H, self.dataset.W).detach().cpu()
                if self.args.model == 'individual':
                    # get from previous subnet.
                    yhat = Image.open(os.path.join(self.result_dir, f'frames_subnet_{int(f/frame_window)}/recon_{f}.png'))
                else:
                    yhat = Image.open(os.path.join(self.result_dir, f'frames_subnet_{w_idx}/recon_{f}.png'))
                yhat = torchvision.transforms.ToTensor()(yhat)

                # PSNR
                psnr_total += PSNR()(yhat, ref).item()
                # SSIM
                ssim_total += SSIM()(torch.unsqueeze(yhat, 0), torch.unsqueeze(ref, 0)).item()
                # LPIPS
                lpips_total += lpips_fn(torch.unsqueeze(yhat, 0), torch.unsqueeze(ref, 0)).item()
            
            psnrs.append(round(psnr_total/ref_full.shape[0], 3))    
            ssims.append(round(ssim_total/ref_full.shape[0], 4))
            lpipss.append(round(lpips_total/ref_full.shape[0], 4))

        with open(os.path.join(self.result_dir, 'PSNR_SSIM_LPIPS.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(psnrs)
            writer.writerow(ssims)
            writer.writerow(lpipss)    

        print(f'PSNR: {psnrs}')
        print(f'SSIM: {ssims}')
        print(f'LPIPS: {lpipss}')


class SDFEval(object):
    def __init__(self, result_dir, dataset, args):
        self.result_dir = result_dir
        self.dataset = dataset
        self.args = args

    def __call__(self, net, w_idx):
        # use best model to create SDF output
        net.load_state_dict(torch.load(os.path.join(self.result_dir, 'model.pth')))
        print("creating ply file...")
        net.select_subnet(w_idx)
        create_mesh(net, os.path.join(self.result_dir, f'recon_{w_idx}'), N = 1024)
        print("ply file created")