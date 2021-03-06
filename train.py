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


# 1) nacitame obrazky z trenovacej mnoziny
# 2) pomocou PCA znizime pocet dimenzii dat
# 3) na upravenych datach spustime SVM a budeme klasifikovat do 4 tried podla natocenia, pouzijeme cross validaciu na najdenie optimalnych parametrov
# 4) vypiseme si PCA maticu do suboru, aby sme ju mohli pouzivat
# 5) spravime funkciu na nacitanie lubovolneho obrazku a jeho preskalovanie a naslednu klasifikaciu
data_folder = './data'
	
def loadTrainData():
	X = np.loadtxt(data_folder + '/images_data', dtype=float)		
	y = np.loadtxt(data_folder + '/rotations', dtype=int)

	simplified_y = []
	#for it in y:
	#	if (it == 0 or it == 180):
	#		simplified_y.append(0)
	#	if (it == 90 or it == 270):
	#		simplified_y.append(1)
	#y = simplified_y

	return X, y

def separateTrainDataByFace(X, y):
	# data pre klasifikator pracujuci s detekciou tvare
	X_face = []
	X_general = []
	# data pre vseobecny klasifikator
	y_face = []
	y_general = []

	for i in range(0, len(X)):
		# posledny atribut urcuje, ci bola detekovana pri niektorom natoceni obrazka tvar
		if (X[i][-1] != -1):
			X_face.append(X[i])
			y_face.append(y[i])

		# odstranime data suvisiace s detekciou tvare, aby neskreslovali pri trenovani
		# lebo hoci je na fotke tvar, stale su to cenne data pre vseobecny klasifikator
		X_general.append(common.stripFaceDetectionData(X[i]))
		y_general.append(y[i])

	X_face = np.array(X_face)
	y_face = np.array(y_face)
	X_general = np.array(X_general)
	y_general = np.array(y_general)

	print "face: " + str(len(X_face)) + " general: " + str(len(X_general))

	return [X_face, X_general], [y_face, y_general]
	#return [X], [y]

def filterFeatures(X, y, feature_count):
	best_feature_selector = SelectKBest(f_classif, k = feature_count)

	X_new = best_feature_selector.fit_transform(X, y)
	return X_new, best_feature_selector

def scale(X):
	data_scaler = StandardScaler().fit(X)
	X_new = data_scaler.fit_transform(X)
	return X_new, data_scaler

def pcaReduction(X, features_count):
	pca = PCA(n_components = features_count)
	pca.fit(random.sample(X, 1000))
	X_new = pca.transform(X)
	return X_new, pca	

def drange(start, stop, step):
	r = start
	while r < stop:
		yield r
		r += step

def trainWithFaceData(X, y):
	X, feature_selector = filterFeatures(X, y, 128)
	X, data_scaler = scale(X)

	joblib.dump(feature_selector, 'saved_models/face_feature_selector.pkl', compress=9)
	joblib.dump(data_scaler, 'saved_models/face_data_scaler.pkl', compress=9)

	#X, pca = pcaReduction(data_transformed, 32)

	sys.stderr.write("Training started\n")	
	best_a = -1
	best_score = 0

	clf = RandomForestClassifier(max_depth=20, n_estimators=256, max_features=10)
	scores = cross_validation.cross_val_score(clf, X, y)	
	clf.fit(X, y)
	joblib.dump(clf, 'saved_models/face_classifier.pkl', compress=9)

	avg_score = scores.mean()
	print "eclf: " + str(avg_score)

def trainGeneral(X, y):
	X, feature_selector = filterFeatures(X, y, 2048)
	X, data_scaler = scale(X)
	X, pca = pcaReduction(X, 32)
	joblib.dump(feature_selector, 'saved_models/general_feature_selector.pkl', compress=9)
	joblib.dump(data_scaler, 'saved_models/general_data_scaler.pkl', compress=9)
	joblib.dump(pca, 'saved_models/general_pca_reducer.pkl', compress=9)


	sys.stderr.write("Training started\n")	
	best_a = -1
	best_score = 0

	for a in drange(0.001, 0.002, 0.0001):
		sys.stderr.write("C: " + str(a) + "\n")# + " gamma: " + str(g) + "r: " + str(c0) + "\n")
		clf = svm.SVC(C=a, kernel='linear')
		scores = cross_validation.cross_val_score(clf, X, y)	
		avg_score = scores.mean()
		print avg_score
		if (avg_score > best_score):
			best_score = avg_score
			best_a = a
	
	clf = svm.SVC(C=best_a, kernel='linear')
	clf.fit(X, y)
	joblib.dump(clf, 'saved_models/general_classifier.pkl', compress=9)

	sys.stderr.write("SVM got on training data cross-validation score " + str(best_score) + "; C=" + str(best_a) + "\n")



print "Loading images..."
images, rotations = loadTrainData()
images_groups, rotations_groups = separateTrainDataByFace(images, rotations)
print "Finished loading"

trainWithFaceData(images_groups[0], rotations_groups[0])
trainGeneral(images_groups[1], rotations_groups[1])
