from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt

def load_data(data_path, label_path):

    # Labels:
    #0: none
    #1: left 
    #2: right 
    #3: reverse
    #4: stop
    #5: goal

    # Load images and labels
    image_files = sorted([os.path.join(data_path, fname) for fname in os.listdir(data_path) if fname.endswith('.png')],
                         key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    
    images = [cv2.imread(img_path) for img_path in image_files]
    labels = np.loadtxt(label_path, dtype=float, delimiter=',')
    labels_sorted = labels[labels[:,0].argsort()][:,1]

    # Check to be sure the images are sorted correctly
    # img_rgb = cv2.cvtColor(images[30], cv2.COLOR_BGR2RGB)
    # plt.imshow(img_rgb)
    # plt.show()

    # Split into training and testing sets
    images_train, images_test, labels_train, labels_test = train_test_split(images, labels_sorted, test_size=0.3)

    # Process training images
    images_train = [preprocess(image, set = 'train') for image in images_train]
    images_test = [preprocess(image, set = 'test') for image in images_test]

    return np.array(images_train), np.array(images_test), np.array(labels_train), np.array(labels_test)

def preprocess(image, set):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32) / 255.0

    if set == 'train':
        # img = cv2.resize(img, (32, 32))
        flip_code = random.choice([-1, 0, 1, None])  # -1: both, 0: vertical, 1: horizontal, None: no flip
        if flip_code is not None:
            img = cv2.flip(img, flip_code)
    
    return img 

def fit_data(images, labels):
    images_flattened = images.reshape(len(images), -1)  # Flatten images into num_images x num_features (pixels)

    # clf = svm.SVC(C=10, gamma=0.001, kernel='rbf', class_weight='balanced')
    # clf = RandomForestClassifier(n_estimators=1000)
    clf = KNeighborsClassifier(n_neighbors=2)
    clf.fit(images_flattened, labels)
    return clf

def test(images, labels):
    labels_predicted = model.predict(images.reshape(len(images), -1))
    acc = accuracy_score(labels, labels_predicted)
    print("Accuracy: ", acc)
    return acc


if __name__ =="__main__":

    data_path = "data/2024F_imgs"
    label_path = "data/2024F_imgs/labels.txt"

    # Load data and fit model
    images_train, images_test, labels_train, labels_test = load_data(data_path, label_path)
    model = fit_data(images_train, labels_train)

    # Run inference
    acc = test(images_test, labels_test)