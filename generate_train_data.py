from PIL import Image
from os import walk
from os import path
from os import makedirs
import random
from skimage import exposure, feature, img_as_float
import numpy as np
import common
# aby to neusekavalo dlhy string pri vypise
np.set_printoptions(threshold=np.nan)


original_images_folder = './original_images'
data_folder = './data'
new_images_folder = data_folder + '/images'
rotations_output_file = open(data_folder + '/rotations', 'w+')
images_data_output_file = open(data_folder + '/images_data', 'w+')

images = []
for (dirpath, dirnames, filenames) in walk(original_images_folder):
    images.extend(filenames)
    break

if not path.exists(new_images_folder):
    makedirs(new_images_folder)
counter = 0
img_data_together = ''
rotations = []
for filename in images:
	print filename
	rotation = random.randint(0,3)
	rotations.append(rotation)
	rotations_output_file.write(str(rotation) + '\n')
	
	img = Image.open(original_images_folder + '/' + filename).rotate(rotation * 90)
	hog_img1, hog_img2, color_histogram = common.preprocess(img)

	img_data = np.append(np.append(hog_img1, hog_img2), color_histogram)
	img_data_together += (' '.join(map(str, img_data)) + '\n')
	counter += 1
	print counter

images_data_output_file.write(img_data_together)
