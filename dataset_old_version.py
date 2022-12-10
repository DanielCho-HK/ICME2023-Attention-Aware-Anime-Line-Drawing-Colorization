from PIL import Image
from torch.utils.data import Dataset
import os


class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None, resize_scale=None):
        super(CustomDataset, self).__init__()
        self.image_dir = image_dir
        self.image_filename = [x for x in sorted(os.listdir(self.image_dir))]
        self.resize_scale = resize_scale
        self.transform = transform

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.image_filename[index])
        img = Image.open(img_path)

        sketch = img.crop((img.width // 2, 0, img.width, img.height))  # sketch
        color_img = img.crop((0, 0, img.width // 2, img.height))       # colorful image

        sketch = sketch.resize((self.resize_scale, self.resize_scale), Image.BILINEAR)
        color_img = color_img.resize((self.resize_scale, self.resize_scale), Image.BILINEAR)

        sketch = self.transform(sketch)
        color_img = self.transform(color_img)

        return sketch, color_img

    def __len__(self):
        return len(self.image_filename)
