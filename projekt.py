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


# 1) nacitame obrazky z trenovacej mnoziny
# 2) pomocou PCA znizime pocet dimenzii dat
# 3) na upravenych datach spustime SVM a budeme klasifikovat do 4 tried podla natocenia, pouzijeme cross validaciu na najdenie optimalnych parametrov
# 4) vypiseme si PCA maticu do suboru, aby sme ju mohli pouzivat
# 5) spravime funkciu na nacitanie lubovolneho obrazku a jeho preskalovanie a naslednu klasifikaciu
data_folder = './data'
feature_count = 512
	
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
	X_face = []
	X_no_face = []
	y_face = []
	y_no_face = []

	for i in range(0, len(X)):
		if (X[i][-1] != -1):
			X_face.append(X[i])
			y_face.append(y[i])
		else:
			X_no_face.append(X[i])
			y_no_face.append(y[i])

	y_face = np.array(y_face)
	y_no_face = np.array(y_no_face)
	X_face = np.array(X_face)
	X_no_face = np.array(X_no_face)
	print "face: " + str(len(X_face)) + " no face: " + str(len(X_no_face))

	#return [X_face, X_no_face], [y_face, y_no_face]
	return [X], [y]

def reduceDimmensionalityOfData(X, y):
	best_feature_selector = SelectKBest(f_classif, k = feature_count)
	data_scaler = StandardScaler().fit(X)

	X_new = best_feature_selector.fit_transform(X, y)
	X_new = data_scaler.fit_transform(X_new)
	return X_new, best_feature_selector, data_scaler

def drange(start, stop, step):
	r = start
	while r < stop:
		yield r
		r += step

def train(X, y):
	sys.stderr.write("Training started\n")	
	best_a = -1
	best_score = 0
	#clf1 = AdaBoostClassifier(svm.SVC(probability=True,kernel='poly', degree=2), n_estimators=1)
	#scores = cross_validation.cross_val_score(clf1, X, y)
	#print scores.mean() 	
	#clf2 = AdaBoostClassifier(svm.SVC(probability=True,kernel='linear'), n_estimators=50)
	#scores = cross_validation.cross_val_score(clf2, X, y)
	#print scores.mean() 	
	#clf3 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=200, learning_rate=0.5)
	#scores = cross_validation.cross_val_score(clf3, X, y)
	#print scores.mean() 	
	clf4 = KNeighborsClassifier(80)
	scores = cross_validation.cross_val_score(clf4, X, y)
	print scores.mean()
	clf5 = RandomForestClassifier(max_depth=20, n_estimators=500, max_features=3)
	scores = cross_validation.cross_val_score(clf5, X, y)
	print scores.mean()
	clf6 = BaggingClassifier(KNeighborsClassifier(80), max_samples=0.7, max_features=0.7)
	scores = cross_validation.cross_val_score(clf6, X, y)	
	avg_score = scores.mean()
	print "eclf: " + str(avg_score)

	for a in drange(0.001, 0.003, 0.0001):
		sys.stderr.write("C: " + str(a) + "\n")# + " gamma: " + str(g) + "r: " + str(c0) + "\n")
		clf = svm.SVC(C=a, kernel='linear')
		scores = cross_validation.cross_val_score(clf, X, y)	
		avg_score = scores.mean()
		print avg_score
		if (avg_score > best_score):
			best_score = avg_score
			best_a = a
	sys.stderr.write("SVM got on training data cross-validation score " + str(best_score) + "; C=" + str(best_a) + "\n")


print "Loading images..."
images, rotations = loadTrainData()
images_groups, rotations_groups = separateTrainDataByFace(images, rotations)
print "Finished loading"
print "Dimmensionality reduction in progress..."

for i in range(0, len(images_groups)):
	data_transformed, feature_selector, data_scaler = reduceDimmensionalityOfData(images_groups[i], rotations_groups[i])
	print "Finished dimmensionality reduction"
	train(data_transformed, rotations_groups[i])
