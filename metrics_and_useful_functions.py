import numpy as np
import matplotlib.pylab as plt
import itertools
import random
from keras.metrics import top_k_categorical_accuracy
from keras import backend as K
from keras.applications.xception import preprocess_input as pp_input_xcept
from keras.applications.resnet50 import preprocess_input as pp_input_rn50



def top_1_categorical_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=1)

def top_3_categorical_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_5_categorical_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)

def precision(y_true, y_pred):
    """
    Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precis = true_positives / (predicted_positives + K.epsilon())
    return precis

def recall(y_true, y_pred):
    """
    Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    rec = true_positives / (possible_positives + K.epsilon())
    return rec

def F1_score(y_true, y_pred):
    '''
    defined as the harmonic average of precision and recall,
    i.e.   2*p*r / (p+r)
    '''
    def precision(y_true, y_pred):
        """
        Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precis = true_positives / (predicted_positives + K.epsilon())
        return precis
    def recall(y_true, y_pred):
        """
        Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        rec = true_positives / (possible_positives + K.epsilon())
        return rec
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    f1 = 2 * ((p * r) / (p + r + K.epsilon()))
    return f1

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def crop_image(img):
    im = img
    if len(im.shape) < 3:  # make black&white images RGB
        im = np.stack((im, im, im), -1)
    if im.shape[2] == 4:  # remove 4th channel if present
        im = im[:, :, 0:3]
    # now random 224x224 crop
    rand_numb_0 = random.randint(0, im.shape[0] - 224)
    rand_numb_1 = random.randint(0, im.shape[1] - 224)
    im = im[rand_numb_0:(rand_numb_0 + 224), rand_numb_1:(rand_numb_1 + 224), :] # random crop
    # add one dimension for prediction
    im = np.expand_dims(im, axis=0)
    return im

def crop_image_web(img):
    im = img
    if len(im.shape) < 3:  # make black&white images RGB
        im = np.stack((im, im, im), -1)
    if im.shape[2] == 4:  # remove 4th channel if present
        im = im[:, :, 0:3]
    # now 224x224 crop at the center
    numb_0 = im.shape[0] // 2 - 112
    numb_1 = im.shape[1] // 2 - 112
    im = im[numb_0:(numb_0 + 224), numb_1:(numb_1 + 224), :]
    # add one dimension for prediction
    im = np.expand_dims(im, axis=0)
    return im

def crop_image_train(img):
    im = img
    if len(im.shape) < 3:  # make black&white images RGB
        im = np.stack((im, im, im), -1)
    if im.shape[2] == 4:  # remove 4th channel if present
        im = im[:, :, 0:3]
    # now random 224x224 crop
    rand_numb_0 = random.randint(0, im.shape[0] - 224)
    rand_numb_1 = random.randint(0, im.shape[1] - 224)
    im = im[rand_numb_0:(rand_numb_0 + 224), rand_numb_1:(rand_numb_1 + 224), :]
    return im

def crop_image_valid(img):
    im = img
    if len(im.shape) < 3:  # make black&white images RGB
        im = np.stack((im, im, im), -1)
    if im.shape[2] == 4:  # remove 4th channel if present
        im = im[:, :, 0:3]
    # now 224x224 crop at the center
    numb_0 = im.shape[0] // 2 - 112
    numb_1 = im.shape[1] // 2 - 112
    im = im[numb_0:(numb_0 + 224), numb_1:(numb_1 + 224), :]
    return im

def crop_image_train_xception(img):
    im = img
    im = pp_input_xcept(im)
    if len(im.shape) < 3:  # make black&white images RGB
        im = np.stack((im, im, im), -1)
    if im.shape[2] == 4:  # remove 4th channel if present
        im = im[:, :, 0:3]
    # now random 224x224 crop
    rand_numb_0 = random.randint(0, im.shape[0] - 224)
    rand_numb_1 = random.randint(0, im.shape[1] - 224)
    im = im[rand_numb_0:(rand_numb_0 + 224), rand_numb_1:(rand_numb_1 + 224), :]
    return im

def crop_image_valid_xception(img):
    im = img
    im = pp_input_xcept(im)
    if len(im.shape) < 3:  # make black&white images RGB
        im = np.stack((im, im, im), -1)
    if im.shape[2] == 4:  # remove 4th channel if present
        im = im[:, :, 0:3]
    # now 224x224 crop at the center
    numb_0 = im.shape[0] // 2 - 112
    numb_1 = im.shape[1] // 2 - 112
    im = im[numb_0:(numb_0 + 224), numb_1:(numb_1 + 224), :]
    return im

def crop_image_train_resnet50(img):
    im = img
    im = pp_input_rn50(im)
    if len(im.shape) < 3:  # make black&white images RGB
        im = np.stack((im, im, im), -1)
    if im.shape[2] == 4:  # remove 4th channel if present
        im = im[:, :, 0:3]
    # now random 224x224 crop
    rand_numb_0 = random.randint(0, im.shape[0] - 224)
    rand_numb_1 = random.randint(0, im.shape[1] - 224)
    im = im[rand_numb_0:(rand_numb_0 + 224), rand_numb_1:(rand_numb_1 + 224), :]
    return im

def crop_image_valid_resnet50(img):
    im = img
    im = pp_input_rn50(im)
    if len(im.shape) < 3:  # make black&white images RGB
        im = np.stack((im, im, im), -1)
    if im.shape[2] == 4:  # remove 4th channel if present
        im = im[:, :, 0:3]
    # now 224x224 crop at the center
    numb_0 = im.shape[0] // 2 - 112
    numb_1 = im.shape[1] // 2 - 112
    im = im[numb_0:(numb_0 + 224), numb_1:(numb_1 + 224), :]
    return im

def crop_train_to_grayscale(img):
    im = img
    if im.shape[2] == 4:  # remove 4th channel if present
        im = im[:, :, 0:3]
    # to grayscale
    def rgb2gray(rgb):
        return np.around(np.dot(rgb[..., :3], [0.299, 0.587, 0.114])).astype('int')
    im = rgb2gray(im)
    im = np.stack((im, im, im), axis=2)
    # now random 224x224 crop
    rand_numb_0 = random.randint(0, im.shape[0] - 224)
    rand_numb_1 = random.randint(0, im.shape[1] - 224)
    im = im[rand_numb_0:(rand_numb_0 + 224), rand_numb_1:(rand_numb_1 + 224), :]
    return im

def crop_valid_to_grayscale(img):
    im = img
    if im.shape[2] == 4:  # remove 4th channel if present
        im = im[:, :, 0:3]
    # to grayscale
    def rgb2gray(rgb):
        return np.around(np.dot(rgb[..., :3], [0.299, 0.587, 0.114])).astype('int')
    im = rgb2gray(im)
    im = np.stack((im, im, im), axis=2)
    # now 224x224 crop at the center
    numb_0 = im.shape[0] // 2 - 112
    numb_1 = im.shape[1] // 2 - 112
    im = im[numb_0:(numb_0 + 224), numb_1:(numb_1 + 224), :]
    return im
