# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 22:14:29 2019

@author: basi9
"""
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.image as mpimg
import random
import sys
from datetime import datetime
# due righe sotto per caricare qualsiasi risoluzione di immagine
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


# creo il dataset contenente i nomi dei file .jpg da leggere e le rispettive labels
all_data_info_true300 = pd.read_csv("C:\\Users\\basi9\\Desktop\\ML Project\\Data\\all_data_info_true300.csv")

all_data_info_true300_count = all_data_info_true300.groupby('artist').count()
print(all_data_info_true300_count.shape)
artist_list = all_data_info_true300_count.index.values.tolist()

## Image processing to get the starting data for training the model
from keras.preprocessing.image import ImageDataGenerator

df = all_data_info_true300.loc[:, ['artist','new_filename']]

train_datagen = ImageDataGenerator(horizontal_flip=True)
valid_datagen = ImageDataGenerator(horizontal_flip=False)
#featurewise_center=True   0-center
#featurewise_std_normalization    normalize

train_generator = train_datagen.flow_from_dataframe(df,\
"C:\\Users\\basi9\\Desktop\\ML Project\\Data\\train", \
target_size=(224, 224), x_col='new_filename',\
y_col='artist', has_ext=True, seed=100)
#Found 13680 images belonging to 57 classes.

valid_generator = valid_datagen.flow_from_dataframe(df,\
"C:\\Users\\basi9\\Desktop\\ML Project\\Data\\valid",\
target_size=(224, 224), x_col='new_filename',\
y_col='artist', has_ext=True, seed=100)
#Found 1710 images belonging to 57 classes.

#color_mode='rgb' default
#has_ext has been deprecated, extensions included
#class_mode= default categorical
#batch_size: size of the batches of data (default: 32)

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
#i.e. 13680//32 = 427
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

##############################################
##########  Transfer Learning with ResNet50 ##
##############################################

### Only for feature extraction as we practically only changed the output classes

from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

n_classes= len(artist_list)
base_model = ResNet50(input_shape=(224,224,3), weights='imagenet',include_top=False)
#
base_model.summary()

x = GlobalAveragePooling2D()(base_model.output)
#not sure about adding flatten
x = Flatten(x)
x = Dropout(0.3)(x)
#put or not keras.layers.?

# let's add a fully-connected layer
#x = keras.layers.Dense(1024, activation='relu')(x)

# we add a logistic/classification layer for our classes
output = Dense(n_classes, activation='softmax')(x)

# this is the model we will train
model50 = Model(inputs=base_model.input, outputs=output)

model50.summary()

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional ResNet50 layers
for layer in base_model.layers:
    layer.trainable = False
#in fact
model50.summary()    

# compile the model (should be done *after* setting layers to non-trainable)
from keras.metrics import top_k_categorical_accuracy
def top_1_categorical_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=1)
def top_3_categorical_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

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
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

from keras.callbacks import ReduceLROnPlateau
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',factor=0.1,\
                                            patience=3,verbose=1,min_lr=0.0001)


model50.compile(optimizer='adam', loss='categorical_crossentropy',\
                metrics=['accuracy', top_1_categorical_accuracy, top_3_categorical_accuracy, precision, recall, F1_score])

time_start = datetime.now()
model50.fit_generator(generator=train_generator,\
                    steps_per_epoch=STEP_SIZE_TRAIN,\
                    validation_data=valid_generator,\
                    validation_steps=STEP_SIZE_VALID,\
                    verbose=2, epochs=30, callbacks=[learning_rate_reduction])
time_end = datetime.now()
print('Tempo di esecuzione per fit_gen: {}'.format(time_end-time_start))

model50.save('data/resnet50_dropTOP_30ep.h5')
# train the model on the new data for a few epochs
##model.fit_generator(generator=training_generator,
#                    validation_data=validation_generator,
#                    use_multiprocessing=True,
#                    workers=3)



####################################################
### FINE-TUNING

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

## let's visualize layer names and layer indices to see how many layers
## we should freeze:
#for i, layer in enumerate(base_model.layers):
#   print(i, layer.name)
#
## we chose to train the top 2 inception blocks, i.e. we will freeze
## the first 249 layers and unfreeze the rest:
#for layer in model.layers[:249]:
#   layer.trainable = False
#for layer in model.layers[249:]:
#   layer.trainable = True
#
## we need to recompile the model for these modifications to take effect
## we use SGD with a low learning rate
#from keras.optimizers import SGD
#model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
#
## we train our model again (this time fine-tuning the top 2 inception blocks
## alongside the top Dense layers
#model.fit_generator(...)
