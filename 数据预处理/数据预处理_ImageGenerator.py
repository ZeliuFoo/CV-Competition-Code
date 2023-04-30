from keras.preprocessing.image import ImageDataGenerator
import cv2 as cv
import numpy as np
import os

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1/255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

data_path = os.getcwd()

for filesname in os.listdir(data_path+'/图片'):
    file_path = os.path.join(data_path+'/图片',filesname)
    img = cv.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    img = np.expand_dims(img,0)

    count = 0
    for batch in datagen.flow(img,batch_size=1,save_to_dir=data_path+'/结果/',save_format='jpg',save_prefix='new_photo'):
        if count == 20:
            break
        else:
            count+=1