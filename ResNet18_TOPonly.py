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
all_data_info_true300 = pd.read_csv("data/all_data_info_true300.csv")

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
"data/train", \
target_size=(224, 224), x_col='new_filename',\
y_col='artist', has_ext=True, seed=100)
#Found 13680 images belonging to 57 classes.

valid_generator = valid_datagen.flow_from_dataframe(df,\
"data/valid",\
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


###   ResNet18 - transfer learning   ###

# transfer learning
#making it doable on GPU
import tensorflow as tf
import keras  
from keras import backend as K
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

# build model
from classification_models import ResNet18
n_classes = len(artist_list)
base_model = ResNet18(input_shape=(224,224,3), weights='imagenet', include_top=False)
x = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation='softmax')(x)
model = keras.models.Model(inputs=[base_model.input], outputs=[output])
model.summary()

# We only train top layers, therefore we freeze the ResNet ones
len(model.layers)
model.layers[(len(model.layers) - 2) : len(model.layers)]
for i in range(len(model.layers) - 2):
    model.layers[i].trainable = False
model.summary()

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
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',\
                                            factor=0.1,\
                                            patience=3,\
                                            verbose=1,\
                                            min_lr=0.0001)
    
# training
model.compile(optimizer='Adam', loss='categorical_crossentropy',\
              metrics=['accuracy', top_1_categorical_accuracy, top_3_categorical_accuracy, precision, recall, F1_score])
model.fit_generator(generator=train_generator,\
                    steps_per_epoch=STEP_SIZE_TRAIN,\
                    validation_data=valid_generator,\
                    validation_steps=STEP_SIZE_VALID,\
                    epochs=5, verbose=2,callbacks=[learning_rate_reduction])
model.save('data/ResNet18_TOPonly_5eps.h5')  # TL sta per 'Transfer Learning'


import pickle
import os
os.system("mkdir dictionaries")
with open('dictionaries/ResNet18_TOPonly_5epochsDict.pkl', 'wb') as file_pi:
    pickle.dump(model.history.history, file_pi)

# list all data in history
print(model.history.history.keys())
# summarize history for accuracy
plt.figure()
plt.plot(model.history.history['acc'])
plt.plot(model.history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show(block=False)

#### Once I saved these infos

#load the model
model18_FineTune = keras.models.load_model('data/ResNet18_TOPonly_5eps.h5',\
custom_objects={'top_3_categorical_accuracy': top_3_categorical_accuracy,\
'precision': precision,'recall': recall,'F1_score': F1_score})

len(model18_FineTune.layers)
model18_FineTune.layers[(len(model18_FineTune.layers) - 2) : len(model18_FineTune.layers)]
for i in range(len(model18_FineTune.layers) - 2):
    model18_FineTune.layers[i].trainable = True
model18_FineTune.summary()

#FineTuning
# training the whole model with previously trained TOP and now with Imagenet starting weights for ResnNet18
model18_FineTune.compile(optimizer='Adam', loss='categorical_crossentropy',\
              metrics=['accuracy', top_1_categorical_accuracy, top_3_categorical_accuracy, precision, recall, F1_score])
model18_FineTune.fit_generator(generator=train_generator,\
                    steps_per_epoch=STEP_SIZE_TRAIN,\
                    validation_data=valid_generator,\
                    validation_steps=STEP_SIZE_VALID,\
                    epochs=25, verbose=2,callbacks=[learning_rate_reduction])
model.save('data/ResNet18_FineTune_25epochs.h5')  

# load the history of the saved model from the pickle file
with open('dictionaries/ResNet18_TOPonly_5epochsDict.pkl', 'rb') as file_pi:
    hist = pickle.load(file_pi)
# merge the two dicts with the histories
hist2 = model.history.history
h = {key:np.hstack([hist[key],hist2[key]]) for key in hist.keys()}

# save new history dict in a new pickle file
with open('dictionaries/ResNet18_FineTune_25epochsDict.pkl', 'wb') as file_pi:
    pickle.dump(h, file_pi)

# accuracy plots
plt.figure()
plt.plot(h['acc'])
plt.plot(h['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show(block=False)

plt.savefig('ResNet18_FineTune.png')
