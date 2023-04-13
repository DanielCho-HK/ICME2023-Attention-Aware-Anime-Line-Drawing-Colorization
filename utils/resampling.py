from PIL import Image

img = Image.open('65.jpg')
print(img)
img_r = img.resize((256, 256), resample=Image.Resampling.LANCZOS)
img_r.save('resize.jpg')
img_re = img_r.resize((480, 660), resample=Image.Resampling.LANCZOS)
img_re.save('restore.jpg')
