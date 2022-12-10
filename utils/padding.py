import os
import cv2


path_name = 'C:\\Users\\user\\Desktop\\fashion illustration\\full\\'

for item in os.listdir(path_name):
    img_path = os.path.join(path_name, item)
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    img = cv2.copyMakeBorder(img, 0, 0, (height-width)//2, (height-width)//2,
                             cv2.BORDER_CONSTANT, value=(255, 255, 255))

    out_dir = './padding'
    cv2.imwrite(os.path.join(out_dir, item), img)








