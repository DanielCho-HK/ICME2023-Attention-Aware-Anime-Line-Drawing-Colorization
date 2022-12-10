from model_our import Generator
from model_SGA import Generator_SGA
from model_SCFT import Generator_SCFT
import torch
from torchvision import transforms, utils
from PIL import Image

sketch = Image.open("./sketch/9866.png")
reference = Image.open("./reference/17606.png")

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

sketch = s_transform(sketch)
reference = r_transform(reference)
sketch = torch.unsqueeze(sketch, 0)
reference = torch.unsqueeze(reference, 0)


gen = Generator_SCFT(sketch_channels=1, reference_channels=3)
ckpt = torch.load("./train_SCFT/ckpt_105001.pth", map_location="cpu")
gen.load_state_dict(ckpt["gen"])
gen.eval()

img_gen = gen(sketch, reference)
utils.save_image(
                    img_gen[0].cpu().data,
                    f"gen.jpg",
                    normalize=True,
                    nrow=1,
                    range=(-1, 1),
                )
