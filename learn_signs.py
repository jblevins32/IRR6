from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib

def load_data(data_path, label_path, test_data_set=True):

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
    num_signs = np.sum(labels[:,1] != 0) # Get number of images that are not none class

    # Test images to see if the preprocessing is good! Set view_preprocess = False below to run entire script without interuption
    if test_data_set:
        correct_color_total = 0
        for test_img in range(len(images)):
            test_img = test_img
            _, correct_color = preprocess(images[test_img], set_type = 'test', view_preprocess = False, img_num = test_img)
            print(f"True label: {labels_sorted[test_img]} for image {test_img}, correct color guessed: {correct_color}")
            correct_color_total += correct_color

        print(f"Correct color percent: {correct_color_total/num_signs}")

    # Split into training and testing sets
    images_train, images_test, labels_train, labels_test = train_test_split(images, labels_sorted, test_size=0.3)

    # Process training images
    images_train, _ = zip(*[preprocess(image, set_type = 'train') for image in images_train])
    images_test, _ = zip(*[preprocess(image, set_type = 'test') for image in images_test])

    return np.array(images_train), np.array(images_test), np.array(labels_train), np.array(labels_test)

def preprocess(image, set_type = 'train', view_preprocess = False, img_num = 0):

    # True colors for accuracy of cropping red=0, green=1, blue=2, none=3
    colors_true = (1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,1,0,0,0,0,0,2,2,3,2,2,2,3,1,1,1,1,2,2,1,2,3,1,1,1,3,2,2,2,2,2,0,0,0,0,3,3,3,0,0,0,0,0,0,1,1,1,1,1,2,2,3,0,0,0,0,0,0,0,0,2,2,2,2,1,1,1,0,0,0,0,2,2,2,2,2,1,2,2,2,1,1,1,1,1,1,1,1,1,1,1,3,3,3,3,1,3,2,2,2,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,2,2,2,1,1,1,1,2,1,2,1,2,1,2,1,2,2,3,3,3,3,3,3,3,3,3,1,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,3,3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,3,3)

    # Separate the colored part from the background
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # RED
    lower_red1 = np.array([0, 165, 165])
    upper_red1 = np.array([20, 235, 235])
    lower_red2 = np.array([160, 165, 165])
    upper_red2 = np.array([180, 255, 255])

    # GREEN
    lower_green = np.array([20, 55, 55])
    upper_green = np.array([100, 255, 255])

    # BLUE
    lower_blue = np.array([100, 60, 30])
    upper_blue = np.array([140, 150, 150])

    # Extract the colored part
    mask_red1 = cv2.inRange(img, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(img, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_green = cv2.inRange(img, lower_green, upper_green)
    mask_blue = cv2.inRange(img, lower_blue, upper_blue)

    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Crop image to area of interest if there is a mask
    masks = [mask_red, mask_green, mask_blue]

    mask_mean_highest = -1 # Use to choose the crop that has the highest ratio of true pixels
    for mask_idx, mask in enumerate(masks):
        img_cropped = crop_image(mask)
    
        # Choose the mask with the highest ratio of true pixels
        mask_mean = np.sum(img_cropped != 0)/(img_cropped.shape[0] * img_cropped.shape[1])
        if mask_mean > mask_mean_highest:
            mask_mean_highest = mask_mean
            chosen_crop_img = mask_idx
            img = img_cropped

    # If mask too small, find contours (smooth shapes)
    # if mask_mean_highest < 0.1:
    #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     new_mask = np.zeros_like(mask)
    #     if contours:
    #             # filter by area to remove noise
    #             min_area = 500  # tune based on your image size
    #             large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    #             if large_contours:
    #                 # Keep all large contours
    #                 cv2.drawContours(new_mask, large_contours, -1, 255, thickness=cv2.FILLED)
    #     mask = new_mask
    #     img = crop_image(mask)


    # If mask too small, blur and find max pixel location which is probably where the sign is
    # if mask_mean_highest < 0.1:
    #     img_saved = img
    #     img_blur = cv2.GaussianBlur(img, ksize=(9,9), sigmaX=0)
    #     idx = np.argmax(img_blur) # Find max pixel intensity as area to focus on
    #     x,y = np.unravel_index(idx, img_blur.shape)

    #     # Crop image around x,y by first finding the quadrant
    #     x_dim = img_blur.shape[0]
    #     y_dim = img_blur.shape[1]
    #     x_dim_img = int(img.shape[1]/2)
    #     y_dim_img = int(img.shape[0]/2)
    #     if x > x_dim/2 and y > y_dim/2:
    #         # crop quadrant 4
    #         img = img[y_dim_img:-1,x_dim_img:-1]
    #     elif x < x_dim/2 and y > y_dim/2:
    #         # crop quadrant 3
    #         img = img[y_dim_img:-1,0:x_dim_img]
    #     elif x < x_dim/2 and y < y_dim/2:
    #         # crop quadrant 2
    #         img = img[0:y_dim_img,0:x_dim_img]
    #     else:
    #         # crop quadrant 1
    #         img = img[0:y_dim_img,x_dim_img:-1]

    #     flag = False

    #     # Now again crop to the white space, but first floor low pixels
    #     img = crop_image(img)

    #     # IF the cropping screws things up
    #     if np.any(np.array(img.shape) == 0):
    #         img = img_saved


    # else: flag = False

    # print(f"Chosen color img: {chosen_crop_img}")
    if chosen_crop_img == colors_true[img_num]:
        correct_color = 1
    else:
        correct_color = 0

    # Resize
    img = cv2.resize(img, (128, 128))

    if view_preprocess:
        fig, ax = plt.subplots(1, 5, figsize=(15, 5))
        ax[0].imshow(mask_red)
        ax[0].set_title('Red Mask')

        ax[1].imshow(mask_green)
        ax[1].set_title('Green Mask')

        ax[2].imshow(mask_blue)
        ax[2].set_title('Blue Mask')

        ax[3].imshow(img, cmap='gray')
        ax[3].set_title('Selected Crop')

        ax[4].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax[4].set_title('Original')

        plt.tight_layout()
        plt.show()  

    return img/255, correct_color

def crop_image(img):
    if np.sum(img != 0) != 0:
        x,y = np.where(img != 0)
        x_min = np.min(x)
        x_max = np.max(x)
        y_min = np.min(y)
        y_max = np.max(y)

        # Crop the image according to the mask
        img_cropped = img[x_min:x_max, y_min:y_max]
        return img_cropped
    else:
        # print("No image to crop")
        return img
    
def fit_data(images, labels, alg='knn'):
    images_flattened = images.reshape(len(images), -1)  # Flatten images into num_images x num_features (pixels)

    # Choose the model
    if alg == 'svm':
        clf = svm.SVC(kernel='rbf', C=2, gamma=0.0001)
    if alg == 'rf':
        clf = RandomForestClassifier(n_estimators=1000)
    if alg == 'knn':
        clf = KNeighborsClassifier(algorithm='brute', n_neighbors=4, n_jobs=-1)

    # Fit the model
    clf.fit(images_flattened, labels)
    return clf

def test(model, images, labels):
    labels_predicted = model.predict(images.reshape(len(images), -1))
    acc = accuracy_score(labels, labels_predicted)
    print("Accuracy: ", acc)
    return acc


if __name__ =="__main__":

    data_path = "data/2024F_imgs"
    label_path = "data/2024F_imgs/labels.txt"

    # Load data and fit model
    images_train, images_test, labels_train, labels_test = load_data(data_path, label_path)
    model = fit_data(images_train, labels_train, alg='svm')
    joblib.dump(model, 'saved_model.pkl')

    # Run inference
    acc = test(model, images_test, labels_test)