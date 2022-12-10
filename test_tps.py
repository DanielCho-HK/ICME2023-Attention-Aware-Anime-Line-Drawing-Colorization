from tps_transformation import tps_transform
from PIL import Image
import numpy as np


reference = Image.open("29785.jpg")
reference = np.array(reference)
reference_tps = tps_transform(reference)
reference_tps = Image.fromarray(reference_tps.astype('uint8'))
reference_tps.save("reference.png")
