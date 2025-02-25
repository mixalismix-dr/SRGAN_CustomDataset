import imquality.brisque as brisque
import cv2
import numpy as np
from skimage import io

img = cv2.imread("img1.png", cv2.IMREAD_COLOR)
ndarray = np.asarray(img)

score = brisque.score(ndarray)
print(score)
