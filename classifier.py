import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from metrics_and_useful_functions import crop_image



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
# show the entire original painting
img = plt.imread(os.path.join(work_dir, 'all_paintings', img_number + '.jpg'))
plt.figure()
plt.imshow(img)

# take a random crop of the image
img = crop_image(img)
plt.figure()
plt.imshow(img[0])



###   Load model and predict author   ###
from metrics_and_useful_functions import top_3_categorical_accuracy, precision, recall, F1_score
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

model = load_model(os.path.join(work_dir, 'RN18_from_scratch_30epoch.h5'),
                         custom_objects={'top_3_categorical_accuracy': top_3_categorical_accuracy,
                                         'precision': precision,
                                         'recall': recall,
                                         'F1_score': F1_score})

# generators
valid_datagen = ImageDataGenerator(horizontal_flip=False)
valid_generator = valid_datagen.flow(img, y=None, batch_size=1, shuffle=False, seed=100)

# predict
valid_generator.reset()
preds = model.predict_generator(generator=valid_generator, steps=len(valid_generator), verbose=0)
top_1_pred = np.argmax(preds, axis=1)
top_3_pred = np.argsort(-preds)[0][:3]

# get top-3 predicted authors
results = [(artist_list[k], round(preds[0,k], 3)) for k in top_3_pred]
print(results)
print('The true author of painting ' + img_number + ' is:   ' +
      df_57artists[df_57artists['new_filename'] == (img_number + '.jpg')].artist.to_string(index=False))
