from PIL import Image
from os import walk
from os import path
from os import makedirs
import random
from skimage import exposure, feature, img_as_float
import numpy as np
import common
import cv2
import sys

# aby to neusekavalo dlhy string pri vypise
np.set_printoptions(threshold=np.nan)


original_images_folder = './original_images'
data_folder = './data'
new_images_folder = data_folder + '/images'
rotations_output_file = open(data_folder + '/rotations', 'w+')
images_data_output_file = open(data_folder + '/images_data', 'w+')

# necha snimku bud v povodnom stave alebo ju otoci o 90 stupnov v smere resp. protismere hodinovych ruciciek
def generateRandomRotation():
	return random.randint(0,3) * 90

images = []
for (dirpath, dirnames, filenames) in walk(original_images_folder):
    images.extend(filenames)
    break

if not path.exists(new_images_folder):
    makedirs(new_images_folder)
counter = 0
img_data_arr = []

for filename in images:
	rotation = generateRandomRotation()	
	img = Image.open(original_images_folder + '/' + filename)
	#if (len(common.face_detect(img)) == 0):	
	img = img.rotate(rotation)	
	rotations_output_file.write(str(rotation) + '\n')	

	hog_1, hog_2, color_histogram = common.preprocess(img)
	img_face_rotation_data, guessed_rotation_by_faces = np.array(common.getFaceCountByRotation(img))

	img_data = np.concatenate((hog_1, hog_2, color_histogram, img_face_rotation_data, [guessed_rotation_by_faces])).ravel()
	img_data_arr.append(img_data)
	img.save(new_images_folder + "/" + filename)
	counter += 1
	print filename
	print img_face_rotation_data
	print guessed_rotation_by_faces
	print >> sys.stderr, str(counter)
	#else:
	#	print filename + " has a face"
	#if counter > 500:
	#	break


np.savetxt(images_data_output_file, img_data_arr)
