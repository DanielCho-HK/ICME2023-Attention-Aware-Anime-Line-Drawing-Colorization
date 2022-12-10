import os
from PIL import Image


path_name = './manga_data/test/reference/'
i = 1
for item in sorted(os.listdir(path_name)):  # 进入到文件夹内，对每个文件进行循环遍历
    os.rename(os.path.join(path_name, item), os.path.join(path_name, (str(i) + '.jpg')))
    i += 1

    # img_path = os.path.join(path_name, item)
    # img = Image.open(img_path)
    # print(img)
    # print(len(img.split()))






