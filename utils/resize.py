import os
from PIL import Image


path_name = './in/'

for item in os.listdir(path_name):
    img_path = os.path.join(path_name, item)
    img = Image.open(img_path)
    # img = img.convert('L')
    img = img.resize((256, 256), Image.ANTIALIAS)
    img.save(os.path.join(path_name, item))

