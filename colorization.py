from model_our import Generator
from model_SGA import Generator_SGA
from model_SCFT import Generator_SCFT
from torchvision import transforms, utils
import torch
import os
from PIL import Image

gen_dir = './gen/'

if not os.path.exists(gen_dir):
    os.makedirs(gen_dir)

sketch = [item for item in sorted(os.listdir("./self_recon/sketch"))]
reference = [item for item in sorted(os.listdir("./self_recon/gt"))]

samples = zip(sketch, reference)
s_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ]
    )

r_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ]
)

gen = Generator_SCFT(sketch_channels=1, reference_channels=3)
ckpt = torch.load("./train_SCFT/ckpt_105001.pth", map_location="cpu")
gen.load_state_dict(ckpt["gen"])
gen.eval()

for s, r in samples:
    sketch = Image.open('./self_recon/sketch/'+s)
    reference = Image.open('./self_recon/gt/'+r)
    sketch = s_transform(sketch)
    reference = r_transform(reference)
    sketch = torch.unsqueeze(sketch, dim=0)
    reference = torch.unsqueeze(reference, dim=0)
    img_gen = gen(sketch, reference)
    utils.save_image(
                    img_gen[0].cpu().data,
                    gen_dir + f"{s.split('.')[0]}.png",
                    normalize=True,
                    nrow=1,
                    range=(-1, 1),
                )





