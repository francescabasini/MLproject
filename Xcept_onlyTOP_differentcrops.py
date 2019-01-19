import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import os
import datetime
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from metrics_and_useful_functions import top_3_categorical_accuracy, crop_image_train_xception, crop_image_valid_xception
from PIL import Image
Image.MAX_IMAGE_PIXELS = None



print('Time start: {}'.format(str(datetime.datetime.now())))

# setting the working directory
work_dir = 'D:\\painter_by_numbers_data\\'   # change this according to the directory you want

# read the dataset with all the information about the images
all_data_info = pd.read_csv(os.path.join(work_dir, 'all_data_info.csv'))
print(all_data_info.shape)

# read the three lists of paintings
paint_list_train = open(os.path.join(work_dir, 'paint_list_train.txt'), 'r').read().split('\n')
paint_list_valid = open(os.path.join(work_dir, 'paint_list_valid.txt'), 'r').read().split('\n')
paint_list_test = open(os.path.join(work_dir, 'paint_list_test.txt'), 'r').read().split('\n')
# read the list of artists
artist_list = open(os.path.join(work_dir, 'artist_list.txt'), 'r').read().split('\n')

# read the dataset containing the 17100 image names and their corresponding author
df = pd.read_csv(os.path.join(work_dir, 'df.csv'))
df_train = pd.read_csv(os.path.join(work_dir, 'df_train.csv'))
df_valid = pd.read_csv(os.path.join(work_dir, 'df_valid.csv'))
df_test = pd.read_csv(os.path.join(work_dir, 'df_test.csv'))
print(df.shape); print(df_train.shape); print(df_valid.shape), print(df_test.shape)



###   Xception training only top   ###
# weights are taken from imagenet
# each time, the input crop for training is new

## data augmentation
train_datagen = ImageDataGenerator(horizontal_flip=True,
                                   preprocessing_function=crop_image_train_xception)
valid_datagen = ImageDataGenerator(horizontal_flip=False,
                                   preprocessing_function=crop_image_valid_xception)
train_generator = train_datagen.flow_from_dataframe(df_train, os.path.join(work_dir, 'all_paintings'),
                    target_size=(224, 224), x_col='new_filename', y_col='artist', has_ext=True, seed=100)
valid_generator = valid_datagen.flow_from_dataframe(df_valid, os.path.join(work_dir, 'all_paintings'),
                    target_size=(224, 224), x_col='new_filename', y_col='artist', has_ext=True, seed=100)
early_stopping = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=4, verbose=1,
                               restore_best_weights=True)
lr_reduction = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=3, verbose=1,
                                 min_lr=0.0001, min_delta=0.01)



###   train model for 20 epochs
import pickle
from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.applications.xception import Xception

n_classes = len(artist_list)
model = Sequential()
model.add(Xception(include_top=False, weights='imagenet', pooling='avg', input_shape=(224, 224, 3)))
model.add(Dense(n_classes, activation='softmax'))
model.layers[0].trainable = False
# compiling
model.compile(optimizer='Adam', loss='categorical_crossentropy',
              metrics=['accuracy', top_3_categorical_accuracy])
print(model.summary())   # ~21mln parameters without TL, 116k with TL

# training
epochs = 20
model.fit_generator(generator=train_generator,
                    steps_per_epoch=train_generator.n // train_generator.batch_size,
                    validation_data=valid_generator,
                    validation_steps=valid_generator.n // valid_generator.batch_size,
                    epochs=epochs, verbose=2,
                    callbacks=[early_stopping, lr_reduction])

# save model
model.save(os.path.join(work_dir, 'Xcept_onlyTOP_differentcrops_10epoch.h5'))
model.save_weights(os.path.join(work_dir, 'Xcept_onlyTOP_differentcrops_10epoch_WEIGHTS.h5'))

# save history in a pickle file
with open(os.path.join(work_dir, 'hist_Xcept_onlyTOP_differentcrops_10epoch'), 'wb') as file_pi:
    pickle.dump(model.history.history, file_pi)

# accuracy plots
plt.figure()
plt.plot(model.history.history['acc'])
plt.plot(model.history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show(block=False)



###   Evaluation of best model (10 epochs)   ###
# load the best model
from keras.models import load_model
best_model = load_model(os.path.join(work_dir, 'Xcept_onlyTOP_differentcrops_10epoch.h5'),
                         custom_objects={'top_3_categorical_accuracy': top_3_categorical_accuracy})
# we have to make the batch size of valid_generator to 1 so that
# it can divide exactly the number of instances in the validation set
test_datagen = ImageDataGenerator(horizontal_flip=False,
                                  preprocessing_function=crop_image_valid_xception)
test_generator = test_datagen.flow_from_dataframe(df_test, os.path.join(work_dir, 'all_paintings'),
                            target_size=(224, 224), x_col='new_filename', y_col='artist', has_ext=True, seed=100,
                            batch_size=1, shuffle=False, class_mode=None)
# class_mode=None is crucial for a dataset of which you want to predict the labels,
# otherwise you'll get some problems with the indices



###   Predictions of test set   ###
# get predictions of test set
test_generator.reset()   # reset is crucial!
preds = best_model.predict_generator(generator=test_generator, steps=len(test_generator), verbose=1)

# get indices (numbers) associated to classes (authors)
predicted_class_indices = np.argmax(preds, axis=1)

# create set of labels in a dictionary
labels = train_generator.class_indices
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]



###   Confusion matrix   ###
# create a dataframe with jpg names and predictions (I can do this since shuffle=False in best_valid_generator)
df_jpg_with_predictions = pd.DataFrame({'new_filename': test_generator.filenames,
                                        'artist_predicted': predictions})

# merge the two dataframes
df_test_with_predictions = pd.merge(df_test, df_jpg_with_predictions, how='inner', on='new_filename', sort=False)

# create arrays for confusion matrix
y_true = np.asarray(df_test_with_predictions['artist'])
y_pred = np.asarray(df_test_with_predictions['artist_predicted'])
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)

## Metrics of the model
# training and validation accuracy
import pickle
with open(os.path.join(work_dir, 'hist_Xcept_onlyTOP_differentcrops_10epoch'), 'rb') as file_pi:
    hist = pickle.load(file_pi)
acc_train = hist['acc'][len(hist['acc'])-1]
print('Training accuracy of best model is: {}'.format(acc_train))
evalu = best_model.evaluate_generator(generator=valid_generator, steps=len(valid_generator))
acc_valid = evalu[1]
print('Validation accuracy of best model is: {}'.format(acc_valid))
# true precision/recall/F1_score of the model
precis = np.mean(np.diag(cm) / np.sum(cm, axis = 1))
print('Precision of best model is: {}'.format(precis))
rec = np.mean(np.diag(cm) / np.sum(cm, axis = 0))
print('Recall of best model is: {}'.format(rec))
f1 = 2 * ((precis * rec) / (precis + rec))
print('F1 Score of best model is: {}'.format(f1))
# top3 training and validation accuracy
acc_top3_train = hist['top_3_categorical_accuracy'][len(hist['acc'])-1]
print('Top-3 training accuracy of best model is: {}'.format(acc_top3_train))
acc_top3_valid = evalu[2]
print('Top-3 validation accuracy of best model is: {}'.format(acc_top3_valid))
# print(np.trace(cm) / cm.sum())   # you should get your validation accuracy, otherwise there is something wrong

# Show confusion matrix in a separate window
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()