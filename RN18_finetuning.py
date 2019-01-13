import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import os
import datetime
import keras
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from metrics_and_useful_functions import top_3_categorical_accuracy



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
print(df.shape)



###   ResNet18 onlyTOP first   ###
# weights are taken from imagenet, we train only top layer for now

## data augmentation
train_datagen = ImageDataGenerator(horizontal_flip=True)
valid_datagen = ImageDataGenerator(horizontal_flip=False)
train_generator = train_datagen.flow_from_dataframe(df, os.path.join(work_dir, 'data\\train'),
                    target_size=(224, 224), x_col='new_filename', y_col='artist', has_ext=True, seed=100)
valid_generator = valid_datagen.flow_from_dataframe(df, os.path.join(work_dir, 'data\\valid'),
                    target_size=(224, 224), x_col='new_filename', y_col='artist', has_ext=True, seed=100)
early_stopping = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=3, verbose=1,
                               restore_best_weights=True)



## build model
from classification_models import ResNet18
n_classes = len(artist_list)
base_model = ResNet18(input_shape=(224,224,3), weights='imagenet', include_top=False)
x = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation='softmax')(x)
model = keras.models.Model(inputs=[base_model.input], outputs=[output])
for i in range(len(model.layers) - 2):
    model.layers[i].trainable = False
print(model.summary())
# compiling
model.compile(optimizer='Adam', loss='categorical_crossentropy',
              metrics=['accuracy', top_3_categorical_accuracy])

# training
epochs = 30
model.fit_generator(generator=train_generator,
                    steps_per_epoch=train_generator.n // train_generator.batch_size,
                    validation_data=valid_generator,
                    validation_steps=valid_generator.n // valid_generator.batch_size,
                    epochs=epochs, verbose=2,
                    callbacks=[early_stopping])

# model and weights save
model.save(os.path.join(work_dir, 'RN18_finetuning_1st_step.h5'))
model.save_weights(os.path.join(work_dir, 'RN18_finetuning_1st_step_WEIGHTS.h5'))

# save history in a pickle file
n_epochs_1st_step = len(model.history.history['loss'])
with open(os.path.join(work_dir, 'hist_RN18_finetuning_1st_step'), 'wb') as file_pi:
    pickle.dump({k:v[:(n_epochs_1st_step - early_stopping.patience)] for k,v in model.history.history.items()}, file_pi)



###   Fine-tuning   ###
# now let all the model adjust to the data
from classification_models import ResNet18
n_classes = len(artist_list)
base_model = ResNet18(input_shape=(224,224,3), weights=None, include_top=False)
x = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation='softmax')(x)
model = keras.models.Model(inputs=[base_model.input], outputs=[output])
model.load_weights(os.path.join(work_dir, 'RN18_finetuning_1st_step_WEIGHTS.h5'))
# compiling, lowering the learning rate
lr_reduction_2nd_step = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=2, verbose=1, min_lr=0.00001)
model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy',
              metrics=['accuracy', top_3_categorical_accuracy])
early_stopping_finetuning = EarlyStopping(monitor='val_acc', min_delta=0.005, patience=3, verbose=1,
                                          restore_best_weights=True)
print(model.summary())

# new generators
train_datagen = ImageDataGenerator(horizontal_flip=True)
valid_datagen = ImageDataGenerator(horizontal_flip=False)
train_generator_finetuning = train_datagen.flow_from_dataframe(df, os.path.join(work_dir, 'data\\train'),
                    target_size=(224, 224), x_col='new_filename', y_col='artist', has_ext=True, seed=100)
valid_generator_finetuning = valid_datagen.flow_from_dataframe(df, os.path.join(work_dir, 'data\\valid'),
                    target_size=(224, 224), x_col='new_filename', y_col='artist', has_ext=True, seed=100)

# training
epochs = 30
model.fit_generator(generator=train_generator_finetuning,
                    steps_per_epoch=train_generator_finetuning.n // train_generator_finetuning.batch_size,
                    validation_data=valid_generator_finetuning,
                    validation_steps=valid_generator_finetuning.n // valid_generator_finetuning.batch_size,
                    epochs=epochs, verbose=2,
                    callbacks=[early_stopping_finetuning, lr_reduction_2nd_step])

# save model
model.save(os.path.join(work_dir, 'RN18_finetuning_2nd_step.h5'))
model.save_weights(os.path.join(work_dir, 'RN18_finetuning_2nd_step_WEIGHTS.h5'))

# load the history of the saved model from the pickle file
with open(os.path.join(work_dir, 'hist_RN18_finetuning_1st_step'), 'rb') as file_pi:
    hist = pickle.load(file_pi)
# merge the two dicts with the histories
hist2 = model.history.history
h = {key: hist[key] + hist2[key] for key in hist}
# save new history dict in a new pickle file
n_epochs_2nd_step = len(model.history.history['loss'])
with open(os.path.join(work_dir, 'hist_RN18_finetuning_2nd_step'), 'wb') as file_pi:
    pickle.dump({k:v[:(n_epochs_1st_step - early_stopping.patience + n_epochs_2nd_step - early_stopping_finetuning.patience)]
                 for k,v in h.items()}, file_pi)

# accuracy plots
plt.figure()
plt.plot(h['acc'])
plt.plot(h['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show(block=False)



###   Evaluation of best model (16 epochs)   ###
# load the best model
from keras.models import load_model
best_model = load_model(os.path.join(work_dir, 'RN18_finetuning_2nd_step.h5'),
                         custom_objects={'top_3_categorical_accuracy': top_3_categorical_accuracy})
# we have to make the batch size of valid_generator to 1 so that
# it can divide exactly the number of instances in the validation set
best_valid_generator = valid_datagen.flow_from_dataframe(df, os.path.join(work_dir, 'data\\valid'),
                            target_size=(224, 224), x_col='new_filename', y_col='artist', has_ext=True, seed=1,
                            batch_size=1, shuffle=False, class_mode=None)
# class_mode=None is crucial for a dataset of which you want to predict the labels,
# otherwise you'll get some problems with the indices



###   Predictions of valid set   ###
# get predictions of valid set
best_valid_generator.reset()   # reset is crucial!
preds = best_model.predict_generator(generator=best_valid_generator, steps=len(best_valid_generator), verbose=1)

# get indices (numbers) associated to classes (authors)
predicted_class_indices = np.argmax(preds, axis=1)

# create set of labels in a dictionary
labels = train_generator.class_indices
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]



###   Confusion matrix   ###
# in the following steps I want to create a dataframe with the predictions and actual authors of the validation set
df_val = df[df['new_filename'].isin(paint_list_valid)]   #.artist
# df_val has the true artist in 'artist' column

# I create a dataframe with jpg names and predictions (I can do this since shuffle=False in best_valid_generator)
df_jpg_with_predictions = pd.DataFrame({'new_filename': best_valid_generator.filenames,
                                        'artist_predicted': predictions})

# merge the two dataframes
df_val_with_predictions = pd.merge(df_val, df_jpg_with_predictions, how='inner', on='new_filename', sort=False)

# create arrays for confusion matrix
y_true = np.asarray(df_val_with_predictions['artist'])
y_pred = np.asarray(df_val_with_predictions['artist_predicted'])
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)

## Metrics of the model
# training and validation accuracy
import pickle
with open(os.path.join(work_dir, 'hist_RN18_finetuning_2nd_step'), 'rb') as file_pi:
    hist = pickle.load(file_pi)
acc_train = hist['acc'][15]
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
acc_top3_train = hist['top_3_categorical_accuracy'][15]
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
