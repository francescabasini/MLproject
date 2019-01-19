import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None



# setting the working directory
work_dir = 'D:\\painter_by_numbers_data\\'   # change this according to the directory you want

# load test set
all_data_info_test = pd.read_csv(os.path.join(work_dir, 'all_data_info_test.csv'))
df_test = pd.read_csv(os.path.join(work_dir, 'df_test.csv'))
#df_57artists = pd.read_csv(os.path.join(work_dir, 'df_57artists.csv'))
paint_list_test = open(os.path.join(work_dir, 'paint_list_test.txt'), 'r').read().split('\n')
#paint_57artists_list = open(os.path.join(work_dir, 'paint_57artists_list.txt'), 'r').read().split('\n')
artist_list = open(os.path.join(work_dir, 'artist_list.txt'), 'r').read().split('\n')

# make the user insert the number of his chosen image
img_number = str(input('Enter the image file number: '))
while (img_number + '.jpg') not in paint_list_test:
    print('Entered painting is not from test set.')
    img_number = str(input('Please enter another image file number: '))



# use flow_from_dataframe
from metrics_and_useful_functions import top_3_categorical_accuracy, crop_image_valid
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

model = load_model(os.path.join(work_dir, 'RN18_finetuning_differentcrops_2nd_step.h5'),   # change according to your saved model name
                   custom_objects={'top_3_categorical_accuracy': top_3_categorical_accuracy})

# generator
test_datagen = ImageDataGenerator(preprocessing_function=crop_image_valid)
df_one_img = df_test[df_test['new_filename'] == str(img_number + '.jpg')].reset_index(drop=True)
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
      df_test[df_test['new_filename'] == (img_number + '.jpg')].artist.to_string(index=False))

# show the original image and the cropped one used in prediction
img = plt.imread(os.path.join(work_dir, 'all_paintings', img_number + '.jpg'))
plt.figure()
plt.imshow(img)
plt.figure()
img_cropped = crop_image_valid(img)
plt.imshow(img_cropped)
