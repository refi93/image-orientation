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
from sklearn.naive_bayes import GaussianNB


# 1) nacitame obrazky z trenovacej mnoziny
# 2) pomocou PCA znizime pocet dimenzii dat
# 3) na upravenych datach spustime SVM a budeme klasifikovat do 4 tried podla natocenia, pouzijeme cross validaciu na najdenie optimalnych parametrov
# 4) vypiseme si PCA maticu do suboru, aby sme ju mohli pouzivat
# 5) spravime funkciu na nacitanie lubovolneho obrazku a jeho preskalovanie a naslednu klasifikaciu
data_folder = './data'
	
def loadTrainData():
	X = []
	with open(data_folder + '/images_data') as openfileobject:
		for line in openfileobject:
			its = np.array([float(x) for x in line.strip().split()])
			X.append(its)
	y = map(int, np.loadtxt(data_folder + '/rotations', delimiter="\n").tolist())
	return np.array(X), np.array(y)

def reduceDimmensionalityOfData(data):
	pca = PCA(n_components=32)
	data = preprocessing.scale(data)
	pca.fit(random.sample(data, 1000))
	data_transformed = [pca.transform(x)[0] for x in data]
	return data_transformed, pca	

def drange(start, stop, step):
	r = start
	while r < stop:
		yield r
		r += step



def train(X, y):
	sys.stderr.write("Training started\n")	
	best_a = -1
	best_score = 0
	clf1 = AdaBoostClassifier(svm.SVC(probability=True,kernel='poly', degree=3), n_estimators=1)
	scores = cross_validation.cross_val_score(clf1, X, y)
	print scores.mean() 	
	clf2 = AdaBoostClassifier(svm.SVC(probability=True,kernel='linear'), n_estimators=1)
	scores = cross_validation.cross_val_score(clf2, X, y)
	print scores.mean() 	
	clf3 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=100, learning_rate=0.2)
	scores = cross_validation.cross_val_score(clf3, X, y)
	print scores.mean() 	
	for a in drange(0.001, 0.05, 0.005):
		sys.stderr.write("C: " + str(a) + "\n")# + " gamma: " + str(g) + "r: " + str(c0) + "\n")
		clf = svm.SVC(C=a, kernel='linear')
		scores = cross_validation.cross_val_score(clf, X, y, cv=20)	
		avg_score = scores.mean()
		print avg_score
		if (avg_score > best_score):
			best_score = avg_score
			best_a = a
	#sys.stderr.write("SVM got on training data cross-validation score " + str(best_score) + "; C=" + str(best_a) + "\n")
	#clf = svm.SVC(C = a)
	#clf.fit(train_set, y)
	#return clf.predict(transformed_test_set)

print "Loading images..."
images, rotations = loadTrainData()
print "Finished loading"
print "Dimmensionality reduction in progress..."
data_transformed, pca = reduceDimmensionalityOfData(images)
print "Finished dimmensionality reduction"
train(data_transformed, rotations)
