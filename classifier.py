import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from metrics_and_useful_functions import crop_image_valid



# setting the working directory
work_dir = 'D:\\painter_by_numbers_data\\'   # change this according to the directory you want

# load dataset of 57 artists
df_57artists = pd.read_csv(os.path.join(work_dir, 'df_57artists.csv'))
paint_57artists_list = open(os.path.join(work_dir, 'paint_57artists_list.txt'), 'r').read().split('\n')
artist_list = open(os.path.join(work_dir, 'artist_list.txt'), 'r').read().split('\n')

# make the user insert the number of his chosen image
img_number = str(input('Enter the image file number: '))
while (img_number + '.jpg') not in paint_57artists_list:
    print('Entered painting is not from one of the 57 artists.')
    img_number = str(input('Please enter another image file number: '))
# # show the entire original painting
# img = plt.imread(os.path.join(work_dir, 'all_paintings', img_number + '.jpg'))
# plt.figure()
# plt.imshow(img)
#
# # take a random crop of the image
# img = crop_image(img)
# plt.figure()
# plt.imshow(img[0])



# ###   Load model and predict author   ###
# from metrics_and_useful_functions import top_3_categorical_accuracy
# from keras.models import load_model
# from keras.preprocessing.image import ImageDataGenerator
#
# model = load_model(os.path.join(work_dir, 'RN18_finetuning_2nd_step.h5'),   # change according to your saved model name
#                    custom_objects={'top_3_categorical_accuracy': top_3_categorical_accuracy})
#
# # generators
# valid_datagen = ImageDataGenerator(horizontal_flip=False)
# valid_generator = valid_datagen.flow(img, y=None, batch_size=1, shuffle=False, seed=100)
#
# # predict
# valid_generator.reset()
# preds = model.predict_generator(generator=valid_generator, steps=len(valid_generator), verbose=0)
# top_1_pred = np.argmax(preds, axis=1)
# top_3_pred = np.argsort(-preds)[0][:3]
#
# # get top-3 predicted authors
# results = [(artist_list[k], round(preds[0,k], 3)) for k in top_3_pred]
# print(results)
# print('The true author of painting ' + img_number + ' is:   ' +
#       df_57artists[df_57artists['new_filename'] == (img_number + '.jpg')].artist.to_string(index=False))



# ### Use predict instead of predict_generator
# from metrics_and_useful_functions import top_3_categorical_accuracy
# from keras.models import load_model
#
# model = load_model(os.path.join(work_dir, 'RN18_finetuning_differentcrops_2nd_step.h5'),   # change according to your saved model name
#                    custom_objects={'top_3_categorical_accuracy': top_3_categorical_accuracy})
#
# preds = model.predict(img, batch_size=None, verbose=0)
# top_1_pred = np.argmax(preds, axis=1)
# top_3_pred = np.argsort(-preds)[0][:3]
#
# # get top-3 predicted authors
# results = [(artist_list[k], round(preds[0,k], 3)) for k in top_3_pred]
# print(results)
# print('The true author of painting ' + img_number + ' is:   ' +
#       df_57artists[df_57artists['new_filename'] == (img_number + '.jpg')].artist.to_string(index=False))



### con flow_from_dataframe
from metrics_and_useful_functions import top_3_categorical_accuracy
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

model = load_model(os.path.join(work_dir, 'RN18_finetuning_differentcrops_2nd_step.h5'),   # change according to your saved model name
                   custom_objects={'top_3_categorical_accuracy': top_3_categorical_accuracy})

# generators
test_datagen = ImageDataGenerator(horizontal_flip=True,
                                  preprocessing_function=crop_image_valid)
df_one_img = df_57artists[df_57artists['new_filename'] == str(img_number + '.jpg')].reset_index(drop=True)
test_generator = test_datagen.flow_from_dataframe(df_one_img, os.path.join(work_dir, 'all_paintings'),
                            target_size=(224, 224), x_col='new_filename', y_col='artist', has_ext=True, seed=100,
                            batch_size=1, shuffle=False, class_mode=None)

# predict
test_generator.reset()
preds = model.predict_generator(generator=test_generator, steps=len(test_generator), verbose=0)
top_1_pred = np.argmax(preds, axis=1)
top_3_pred = np.argsort(-preds)[0][:3]

# get top-3 predicted authors
results = [(artist_list[k], round(preds[0,k], 3)) for k in top_3_pred]
print(results)
print('The true author of painting ' + img_number + ' is:   ' +
      df_57artists[df_57artists['new_filename'] == (img_number + '.jpg')].artist.to_string(index=False))

# show the original image and the cropped one used in training
img = plt.imread(os.path.join(work_dir, 'all_paintings', img_number + '.jpg'))
plt.figure()
plt.imshow(img)
plt.figure()
plt.imshow(crop_image_valid(img))
