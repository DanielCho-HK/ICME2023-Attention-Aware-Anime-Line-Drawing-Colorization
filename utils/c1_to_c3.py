import os
import cv2


path_name = './manga/'
i = 1
for item in os.listdir(path_name):  # 进入到文件夹内，对每个文件进行循环遍历
    img_path = os.path.join(path_name, item)
    img = cv2.imread(img_path)
    cv2.imwrite(os.path.join('out', item), img)





