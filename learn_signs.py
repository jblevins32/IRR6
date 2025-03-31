from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cv2
import numpy as np
import os

def load_data(data_path, label_path):
    # LAbeling:
    #0: left arrow
    #1: right arrow or left arched arrow
    #2: right arched arrow
    # Load images and labels
    image_files = sorted([os.path.join(data_path, fname) for fname in os.listdir(data_path) if fname.endswith('.png')])
    images = [cv2.imread(img_path) for img_path in image_files]
    labels = np.loadtxt(label_path, dtype=float, delimiter=',')
    labels_sorted = np.sort(labels,axis=0)[:,1]

    # Split into training and testing sets
    images_train, images_test, labels_train, labels_test = train_test_split(images, labels_sorted, test_size=0.2)

    return np.array(images_train), np.array(images_test), np.array(labels_train), np.array(labels_test)

def fit_data(images, labels):
    images_flattened = images.reshape(len(images), -1)  # Flatten images into num_images x num_features (pixels)

    clf = svm.SVC()
    clf.fit(images_flattened, labels)
    return clf

if __name__ =="__main__":

    data_path = "data/2024F_imgs"
    label_path = "data/2024F_imgs/labels.txt"

    # Load data and fit model
    images_train, images_test, labels_train, labels_test = load_data(data_path, label_path)
    model = fit_data(images_train, labels_train)

    # Run inference
    labels_predicted = model.predict(images_test.reshape(len(images_test), -1))
    acc = accuracy_score(labels_test, labels_predicted)
    print(acc)