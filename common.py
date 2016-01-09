from PIL import Image
from os import walk
from os import path
from os import makedirs
import random
from skimage import exposure, feature, img_as_float
import numpy as np
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage import feature
import cv2
from resizeimage import resizeimage


# subor, kde su ulozene udaje potrebne na detekciu tvari
faceCascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(faceCascPath)

def imgToGray(img):
	return img.convert('L')

def imgResizeAndKeepRatio(img, shorter_side_size):
	width, height = img.size
	if (width > height):
		width = int(shorter_side_size * 1.0/ height * width)
		height = shorter_side_size
	else:
		height = int(shorter_side_size * 1.0 / width * height)
		width = shorter_side_size
	return img.resize((width, height),Image.ANTIALIAS)

def imgToGrayAndResize(img, size):
	return img.convert('L').resize((size, size), Image.ANTIALIAS)

def face_detect(img):
	img = np.array(imgResizeAndKeepRatio(imgToGray(img), 512))
	faces = faceCascade.detectMultiScale(
		img,
		scaleFactor=1.1,
		minNeighbors=10,
		minSize=(20, 20),
		flags = cv2.CASCADE_SCALE_IMAGE
	)
	return faces

# funkcia v pripade, ze na obrazku img najde tvare, vrati rotaciu img vzhladom na spravne zrotovany obrazok
# v pripade, ze nenajde ziadne tvare na obrazku, vrati -1 a do hry prichadzaju dalsie klasifikatory
def getFaceCountByRotation(img):
	max_face_count = 0
	rotation = -1
	face_counts = []
	for i in range(0,4):
		face_count = len(face_detect(img.rotate(i * 90)))
		face_counts.append(face_count)
		
		if (face_count > max_face_count):
			max_face_count = face_count
			rotation = ((4 - i) % 4) * 90 # lebo vraciame, ze ako je ten obrazok zrotovany oproti tomu spravne zrotovanemu
	
	return face_counts, rotation

def vectorBlockshaped(arr, nrows, ncols):
	"""
	Return an array of shape (n, nrows, ncols) where
	n * nrows * ncols = arr.size

	If arr is a 2D array, the returned array should look like n subblocks with
	each subblock preserving the "physical" layout of arr.
	"""
	h, w = arr.shape
	pom = (arr.reshape(h//nrows, nrows, -1, ncols)
				.swapaxes(1,2)
				.reshape(-1, nrows, ncols))
	ret = []	
	for it in pom:
		ret.append(it.ravel())
	return np.array(ret)
	return pom

def arrayThreshold(arr2d, threshold):
	result = []
	for r in range(0, len(arr2d)):
		result.append([])
		for c in range(0, len(arr2d[0])):
			if arr2d[r][c] <= threshold:
				result[r].append(0)
			else:
				result[r].append(arr2d[r][c])
	return result

# zdroj: http://stackoverflow.com/questions/6574782/how-to-whiten-matrix-in-pca
def whiten(X, fudge=1E-18):
	# the matrix X should be observations-by-components
	# get the covariance matrix
	Xcov = np.dot(X.T,X)
	# eigenvalue decomposition of the covariance matrix
	d, V = np.linalg.eigh(Xcov)
	# a fudge factor can be used so that eigenvectors associated with
	# small eigenvalues do not get overamplified.
	D = np.diag(1. / np.sqrt(d+fudge))
	# whitening matrix
	W = np.dot(np.dot(V, D), V.T)
	# multiply by the whitening matrix
	X_white = np.dot(X, W)

	return X_white, W

# konverzia do greyscale a na urcitu velkost a na 1D vektor, nech sa s tym pekne robi
def preprocess(img):
	img = img.convert('L')
	bw_img = imgToGrayAndResize(img, 512)
	scharr_image = arrayThreshold(scharr(bw_img), 0.04) # ocistime obrazok od slabych pixlov, aby sa trebars dal lahsie spoznat horizont
	#canny_image = feature.canny(np.array(bw_img))
	hog1 = feature.hog(scharr_image, orientations=16, pixels_per_cell=(32, 32), cells_per_block=(1, 1), normalise = True, visualise=False)	
	hog2 = []#feature.hog(sobel_image, orientations=20, pixels_per_cell=(32, 32), cells_per_block=(1, 1), normalise = True, visualise=False)
	
	image_blocks = vectorBlockshaped(np.array(bw_img), 128, 128)
	blocks_color_histogram = []
	counter = 0
	for block in image_blocks:
		hist, bin_edges = np.histogram(block, bins=range(0,256))
		blocks_color_histogram.append(hist)

	blocks_color_histogram = np.array(blocks_color_histogram).ravel()
		
	return hog1, hog2, blocks_color_histogram

