from keras import layers
import numpy as np
from matplotlib import pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
import tensorflow as tf


train_dir = 'D:\文献\比赛\模型构建与训练\output/train'
val_dir = 'D:\文献\比赛\模型构建与训练\output/val'
test_dir = 'D:\文献\比赛\模型构建与训练\output/test'

train_datagen = ImageDataGenerator(rescale=1./255,            #归一化
                                   zoom_range=0.15,           #随即缩放的范围
                                   width_shift_range=0.2,     #水平偏移
                                   height_shift_range=0.2,    #垂直偏移
                                   shear_range=0.15,          #随机错切变换的角度
                                   rotation_range=20)         #旋转角度

# #train_datagen = ImageDataGenerator(rescale=1. / 255,          #归一化
#                                    rotation_range=10,         #旋转角度
#                                    width_shift_range=0.1,     #水平偏移
#                                    height_shift_range=0.1,    #垂直偏移
#                                    shear_range=0.1,           #随机错切变换的角度
#                                    zoom_range=0.1,            #随机缩放的范围
#                                    horizontal_flip=False,     #随机将一半图像水平翻转
#                                    fill_mode='nearest')       #填充像素的方法


val_datagen = ImageDataGenerator(rescale=1./255)
# 归一化   datagen训练数据生成

test_datagen = ImageDataGenerator(rescale=1./255)
# 归一化

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224,224),
        batch_size=64,
        seed=2077,
        shuffle=True)

validation_generator = val_datagen.flow_from_directory(
    val_dir,target_size=(224,224),batch_size=64,shuffle=False,seed=2077) # set as validation data

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(224,224),batch_size=1,shuffle=False,seed=2077) # set as validation data

model = tf.keras.Sequential(
        [
            layers.Conv2D(16, (5, 5), activation="relu", input_shape=(224, 224, 3)),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (5, 5), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.BatchNormalization(),
            layers.Conv2D(64, (5, 5), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.BatchNormalization(),
            layers.Conv2D(128, (5, 5), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.BatchNormalization(),
            layers.Conv2D(256, (5, 5), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.BatchNormalization(),
            layers.GlobalAvgPool2D(),
            layers.Dense(512,activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )

model.summary()

model.compile(loss="categorical_crossentropy", optimizer='adam', metrics="accuracy")

history = model.fit(
        train_generator,
        epochs=10,
        workers=8,
        validation_data=validation_generator)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()