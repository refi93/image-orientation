import numpy as np
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler  
from sklearn import svm
from sklearn import cross_validation
from sklearn import preprocessing
from PIL import Image
from os import walk
from os import path
from os import makedirs
import random
from skimage.feature import hog
from skimage import data, color, exposure
import common
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
np.set_printoptions(threshold=np.nan)
import cv2
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

def hasFace(x):
	return x[-1] != -1

def loadFaceClassifier():
	feature_selector = joblib.load(common.abs_path + '/saved_models/face_feature_selector.pkl')
	data_scaler = joblib.load(common.abs_path + '/saved_models/face_data_scaler.pkl')
	clf = joblib.load(common.abs_path + '/saved_models/face_classifier.pkl')
	return feature_selector, data_scaler, clf

def loadGeneralClassifier():
	feature_selector = joblib.load(common.abs_path + '/saved_models/general_feature_selector.pkl')
	data_scaler = joblib.load(common.abs_path + '/saved_models/general_data_scaler.pkl')
	pca = joblib.load(common.abs_path + '/saved_models/general_pca_reducer.pkl')
	clf = joblib.load(common.abs_path + '/saved_models/general_classifier.pkl')

	return feature_selector, data_scaler, pca, clf

def predict(x):
	if hasFace(x):
		feature_selector, data_scaler, clf = loadFaceClassifier()
		x = data_scaler.transform(feature_selector.transform(x))
		return clf.predict(x)[0]
	else:
		feature_selector, data_scaler, pca, clf = loadGeneralClassifier()
		
		x = common.stripFaceDetectionData(x)
		# selekcia vlastnosti a naskalovanie
		x = data_scaler.transform(feature_selector.transform(x))
		# pca redukcia dimenzii
		x = pca.transform(x)
		return clf.predict(x)[0]


# Get user supplied values
imagePath = sys.argv[1]
img = Image.open(imagePath)

hog_1, color_histogram, img_face_rotation_data, guessed_rotation_by_faces = common.preprocess(img)
x = np.concatenate((hog_1, color_histogram, img_face_rotation_data, [guessed_rotation_by_faces])).ravel()
print predict(x)
