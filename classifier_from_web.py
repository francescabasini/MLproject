import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None



# setting the working directory
work_dir = 'D:\\painter_by_numbers_data\\'   # change this according to the directory you want
artist_list = open(os.path.join(work_dir, 'artist_list.txt'), 'r').read().split('\n')

# let the user insert the new image downloaded in the folder 'new_images_from_web'
while True:
    try:
        img_name = input('Enter the image file name: ')
        img = plt.imread(os.path.join(work_dir, 'new_images_from_web\\', img_name + '.jpg'))
        break
    except FileNotFoundError:
        print('Entered image name does not exist.')



# use flow
from metrics_and_useful_functions import top_3_categorical_accuracy, crop_image_valid
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

model = load_model(os.path.join(work_dir, 'RN18_finetuning_differentcrops_2nd_step.h5'),   # change according to your saved model name
                   custom_objects={'top_3_categorical_accuracy': top_3_categorical_accuracy})

# generator
test_datagen = ImageDataGenerator(preprocessing_function=crop_image_valid)
df_web_img = pd.DataFrame({'new_filename': [img_name + '.jpg'], 'artist': ['xxx']})
test_generator = test_datagen.flow_from_dataframe(df_web_img, os.path.join(work_dir, 'new_images_from_web'),
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

plt.figure()
plt.imshow(img)
plt.figure()
img_cropped = crop_image_valid(img)
plt.imshow(img_cropped)
