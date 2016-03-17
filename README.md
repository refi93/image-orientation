# Machine learning based determination of image orientation

Python implementation of classifiers to determine the right image orientation.

We used mainly SVM and PCA dimmensionality reduction. The images were transformed by sobel edge detection and then we computed their histogram of gradients, we also splitted the original image into 4x4 squares and computed the color histogram for each of them. We also used face detection (Haar cascades) implemented in the opencv library. We gained accuracy 60% in general case on our set of images and on images, where we detected one or more faces, the accuracy was more than 80%.

# Poznamky
subor [generate_train_data.py](https://github.com/refi93/image-orientation/blob/master/generate_train_data.py) predpoklada pritomnost zlozky original_images a v nej spravne orientovane fotografie
Po jeho spusteni sa zacnu iterovat fotografie v zlozke a jedna po druhej spracovavat do dat, ktore budu vyuzite pri trenovani.

O trenovanie sa stara subor [train.py](https://github.com/refi93/image-orientation/blob/master/train.py). Najde optimalne modely pre klasifikatory a ulozi ich do zlozky saved_models.

Klasifikacia potom prebieha tak, ze spustime subor [predict.py](https://github.com/refi93/image-orientation/blob/master/predict.py) a ako parameter mu dame cestu k fotografii, o ktorej chceme vediet jej orientaciu. Predtym ale musime spustit trenovanie, resp. mat v zlozke saved_models ulozene prislusne klasifikatory.
