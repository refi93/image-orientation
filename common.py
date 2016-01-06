from PIL import Image
from os import walk
from os import path
from os import makedirs
import random
from skimage import exposure, feature, img_as_float
import numpy as np
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage import feature

# konverzia do greyscale a na velkost 256x256 a na 1D vektor, nech sa s tym pekne robi
def preprocess(img):
	bw_img = img.convert('L').resize((512, 512), Image.ANTIALIAS)
	canny_image = feature.canny(np.array(bw_img), sigma=1.2)
	hog1 = feature.hog(canny_image, orientations=20, pixels_per_cell=(16, 16), cells_per_block=(1, 1), normalise = True, visualise=False)	
	sobel_image = sobel(bw_img)
	hog2 = feature.hog(sobel_image, orientations=20, pixels_per_cell=(16, 16), cells_per_block=(1, 1), normalise = True, visualise=False)	
	
	avg_colors = np.array(bw_img.resize((16, 16))).ravel()
	return hog1, hog2, avg_colors


