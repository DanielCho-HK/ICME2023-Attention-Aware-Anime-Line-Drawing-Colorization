import os
import torch
from PIL import Image
from torchvision import transforms

transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))


psnr = PSNR()

gt = [item for item in sorted(os.listdir('./self_recon/gt/'))]
gen = [item for item in sorted(os.listdir('./self_recon/gen/'))]
samples = zip(gt, gen)
score = torch.tensor(0.)
for gt_name, gen_name in samples:
    img1 = Image.open('./self_recon/gt/'+gt_name)
    img2 = Image.open('./self_recon/gen/'+gen_name)
    img1 = transform(img1)
    img2 = transform(img2)
    img1 = img1 * 255
    img2 = img2 * 255
    img1 = torch.unsqueeze(img1, dim=0)
    img2 = torch.unsqueeze(img2, dim=0)
    psnr_val = psnr(img1, img2)
    score = score + psnr_val
print(score / len(gt))
