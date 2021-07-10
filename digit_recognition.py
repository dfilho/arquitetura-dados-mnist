import matplotlib.pyplot as plt
import cv2 as cv
from numpy.core.fromnumeric import ndim
import pandas as pd
import numpy as np
from scipy.ndimage.filters import gaussian_filter, median_filter
from sklearn import svm, metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from skimage.util import random_noise

 
def digit_recogntion():
    train = pd.read_csv('input/mnist_train.csv')
    test = pd.read_csv('input/mnist_test.csv')

    #trainnum=5000
    #testnum=5000
    
    #x_train = train.iloc[:trainnum,1:].values
    #y_train = train.iloc[:trainnum,0].values

    #x_test = train.iloc[trainnum:trainnum+testnum,1:].values
    #y_test = train.iloc[trainnum:trainnum+testnum,0].values

    x_train = train.iloc[:,1:].values
    y_train = train.iloc[:,0].values

    x_test = test.iloc[:,1:].values
    y_test = test.iloc[:,0].values

    #return
    #x_train_st = standardization(x_train)
    x_train_deskew = deskew_image_array(x_train)
    #x_train_hog_descriptor = hog_descriptor(x_train)
    #x_train_gaussian_image = gaussian_blur_image_array(x_train)
    #x_train_binarized = numeric_array_binarization(x_train)
    #x_train_canny_image = canny_image_array(x_train_binarized) -- Canny diminuiu muito a acuracia
    #x_train_median_image_blur = median_image_blur_array(x_train)
    #x_train_sharpening_image_array = sharpening_image_array(x_train)
    #x_train_binarized = numeric_array_binarization(x_train_median_image_blur)
    #x_train_dilated_image = dilation_image_array(x_train_binarized) -- Não compensa usar
    x_train_st = standardization(x_train_deskew)

    #plot_numbers(x_train_binarized)
    #return
    #x_test_st = standardization(x_train)
    x_test_deskew = deskew_image_array(x_test)
    #x_test_hog_descriptor = hog_descriptor(x_train)
    #x_test_gaussian_image = gaussian_blur_image_array(x_test)
    #x_test_binarized = numeric_array_binarization(x_test)
    #x_test_canny_image = canny_image_array(x_test_binarized) -- Canny diminuiu muito a acuracia
    #x_test_median_image_blur = median_image_blur_array(x_test)
    #x_test_sharpening_image_array = sharpening_image_array(x_test)
    #x_test_binarized = numeric_array_binarization(x_test_median_image_blur)
    #x_test_dilated_image = dilation_image_array(x_test_binarized) -- Não compensa usar
    x_test_st = standardization(x_test_deskew)

    #plot_numbers(x_train)
    #plot_numbers(x_train_median_image_blur)
    # plt.imshow(x_train[0].reshape(28,28), cmap='gray')
    # plt.show()
    # plt.imshow(x_train_median_image_blur[0].reshape(28,28), cmap='gray')
    # plt.show()

    classification(x_train_st, y_train, x_test_st, y_test)

def dilation_image_array(numeric_array) -> np.ndarray:
    dilation_array = []
    for row in numeric_array:
        kernel = np.ones((28,28),np.uint8)
        erosion = cv.erode(row.reshape(28,28).astype(np.uint8),None)
        dilation = cv.dilate(erosion,None)
        dilation_array.append(dilation.reshape(784,))
    return dilation_array

def deskew_image_array(numeric_array) -> np.ndarray:
    deskew_array = []
    SZ = 28 # images are SZ x SZ grayscale
    affine_flags = cv.WARP_INVERSE_MAP|cv.INTER_LINEAR
    for row in numeric_array:
        m = cv.moments(row.reshape(28,28).astype(np.uint8))
        if abs(m['mu02']) < 1e-2:
            deskew_array.append(row.reshape(784,).astype(np.uint8).copy())
            break
        skew = m['mu11']/m['mu02']
        M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
        deskew_img = cv.warpAffine(row.reshape(28,28).astype(np.uint8),M,(SZ, SZ),flags=affine_flags)
        deskew_array.append(deskew_img.reshape(784,))
    # plt.imshow(numeric_array[0].reshape(28,28), cmap='gray')
    # plt.show()
    # plt.imshow(deskew_array[0].reshape(28,28), cmap='gray')
    # plt.show()
    return deskew_array

def hog_descriptor(numeric_array) -> np.ndarray:
    hog_array = []
    
    winSize = (20,20)
    blockSize = (10,10)
    blockStride = (5,5)
    cellSize = (10,10)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradients = True
    hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradients)
    
    for row in numeric_array:
        descriptor_hog_img = hog.compute(row.reshape(28,28).astype(np.uint8))
        hog_array.append(descriptor_hog_img.reshape(81,))

    return hog_array

def sharpening_image_array(numeric_array) -> np.ndarray:
    sharpening_array = []
    for row in numeric_array:
        filter = np.array([[-1, -1, -1], [-1, 13, -1], [-1, -1, -1]])
        sharpen_img = cv.filter2D(row.reshape(28,28).astype(np.uint8),-1,filter)
        sharpening_array.append(sharpen_img.reshape(784,))
    return sharpening_array

def canny_image_array(numeric_array) -> np.ndarray:
    canny_array = []
    for row in numeric_array:
        canny = cv.Canny(row.reshape(28,28).astype(np.uint8),100,200)
        canny_array.append(canny.reshape(784,))
    return canny_array

def gaussian_blur_image_array(numeric_array) -> np.ndarray:
    gaussian_blur_array = []
    for row in numeric_array:
        gaussian_image = cv.GaussianBlur(row.reshape(28,28).astype(np.uint8), (1,1), 0)
        gaussian_blur_array.append(gaussian_image.reshape(784,))
    return gaussian_blur_array

def median_image_blur_array(numeric_array) -> np.ndarray:
    median_blur_array = []
    for row in numeric_array:
        median_image = cv.medianBlur(row.reshape(28, 28).astype(np.uint8), 3)
        median_blur_array.append(median_image.reshape(784,))
    return median_blur_array

def normalize_image_array(numeric_array) -> np.ndarray:
    normalized_array = []
    for row in numeric_array:
        row = row.astype('float32')
        row /= 255.0
        normalized_array.append(row)
    return normalized_array

def standardization(data_train) -> np.ndarray:
    
    data_train_st = []

    # Divide elements
    for row in data_train:
        Xtrn = np.divide(row, 255)

        # Calculate means
        Xmean = Xtrn.mean(axis=0)
        
        # Standard Deviation
        Xstd = Xtrn.std()

        # Substract means
        # Xtrn_nm = Xtrn - Xmean
        # Xtst_nm = Xtst - Xmean

        # Standardization
        Xtrn_st = (Xtrn - Xmean) / Xstd
        #Xtst_st = (Xtst - Xmean) / Xstd

        data_train_st.append(Xtrn_st.reshape(784,))

    return data_train_st

'''retirado de https://note.nkmk.me/en/python-numpy-opencv-image-binarization/'''
def numeric_array_binarization(numeric_array) -> np.ndarray:
    maxval = 255
    tresh = 64
    binarized = []
    for row in numeric_array:
        ret, im_th = cv.threshold(row.reshape(28, 28).astype(np.uint8), 64, 255, cv.THRESH_BINARY)
        #row = (row > tresh) * maxval
        binarized.append(im_th.reshape(784,))
    return binarized
     

def plot_numbers(number_array, n = 5):
    fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(12, 12))
    for i in range(n**2):
        ax = axs[i // n, i % n]
        ax.imshow(number_array[i].reshape(28, 28), cmap=plt.cm.gray)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
def classification(x_train, y_train, x_test, y_test):
    #param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [ 1, 0.1, 0.01, 0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
    #grid = GridSearchCV(svm.SVC(),param_grid, refit=True,verbose=2)
    #### Fazer o SVM ####
    #grid = svm.SVC(C=1,kernel='rbf',gamma=0.01, cache_size=8000,probability=False)
    grid = svm.SVC()
    # clf.fit(x_train, y_train)
    grid.fit(x_train, y_train)
    expected = y_test

    #predicted = clf.predict(x_test)
    predicted = grid.predict(x_test)

    # print("Classification report for classifier %s:\n%s\n"
    #    % (clf, metrics.classification_report(expected, predicted)))
    print("Classification report for classifier %s:\n%s\n"
        % (grid, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

    #plot_confusion_matrix(clf, x_test, y_test)
    plot_confusion_matrix(grid, x_test, y_test)

    plt.show()


if __name__ == "__main__":
    digit_recogntion()
