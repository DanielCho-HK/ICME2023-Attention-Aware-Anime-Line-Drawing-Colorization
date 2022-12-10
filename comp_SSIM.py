import os
import torch
from PIL import Image
from torchvision import transforms
from pytorch_msssim import ms_ssim, MS_SSIM

transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

gt = [item for item in sorted(os.listdir('./self_recon/gt/'))]
gen = [item for item in sorted(os.listdir('./self_recon/gen_scft/'))]
samples = zip(gt, gen)
score = torch.tensor(0.)
for gt_name, gen_name in samples:
    img1 = Image.open('./self_recon/gt/'+gt_name)
    img2 = Image.open('./self_recon/gen_scft/'+gen_name)
    img1 = transform(img1)
    img2 = transform(img2)
    img1 = img1 * 255
    img2 = img2 * 255
    img1 = torch.unsqueeze(img1, dim=0)
    img2 = torch.unsqueeze(img2, dim=0)
    ms_ssim_val = ms_ssim(img1, img2, data_range=255, size_average=False)
    score = score + ms_ssim_val
print(score / len(gt))















