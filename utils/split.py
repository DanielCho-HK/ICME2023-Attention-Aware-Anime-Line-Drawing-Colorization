import os
import cv2

path_name = './data'


for item in os.listdir(path_name):
    img_path = os.path.join(path_name, item)
    img = cv2.imread(img_path)
    height, width, _ = img.shape

    part1 = img[0:height, 0:352]
    part2 = img[0:height, 352:]

    out_dir1 = './part1'
    out_dir2 = './part2'
    cv2.imwrite(os.path.join(out_dir1, item), part1)
    cv2.imwrite(os.path.join(out_dir2, item), part2)








