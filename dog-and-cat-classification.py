#tensorflow 2.0
from keras.preprocessing.image import ImageDataGenerator , load_img
from keras.utils import to_categorical
#신경망 모델 구축
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from sklearn.model_selection import train_test_split
import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#데이터 저장 확인
#print(os.listdir("./data"))

#global variables, 가로, 세로, 사이즈, RGB 채널 정의
FAST_RUN = False
IMAGE_W = 128
IMAGE_H = 128
IMAGE_SIZE = (IMAGE_W, IMAGE_H)
IMAGE_CHANNELS = 3

#train data set 준비, ('dog', 1), ('cat', 0)
filenames = os.listdir("./data/training_set")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

train_frame = pd.DataFrame({
        'filename' : filenames,
        'category' : categories
})

#test data set 준비, ('dog', 1), ('cat', 0)
filenames = os.listdir("./data/test_set")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

test_frame = pd.DataFrame({
        'filename' : filenames,
        'category' : categories
})
# 데이터 저장 확인
#print(test_frame)

#데이터 밸런스 확인, 분류 문제는 고루 분포가 되어야 함
# test_frame['category'].value_counts().plot.bar()

#sample data
# sample_data = random.choice(filenames) #filenames에서 랜덤 선택
# image = load_img("./data/training_set/" + sample_data)
# plt.imshow(image)

#신경망 구축
model = Sequential()

model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (IMAGE_W, IMAGE_H, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax')) # dense 2, output dog and cat
#모델 컴파일해서 loss 정해줌, categorical_crossentropy
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()

#콜백
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

#Early stop, 20epoch 지나서 정확도가 떨어지면 그만하라는말
earlystop = EarlyStopping(patience=20)

#Learningrate 조절
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                           patience=2,
                                           verbose=1,
                                           factor=0.5,
                                           min_lr=0.00001)
#콜백에 클래스 감기
callbacks = [earlystop, learning_rate_reduction]

# data string화, one hot 인코딩 위해서
train_frame["category"] = train_frame["category"].replace({0 : 'cat', 1: 'dog'})
test_frame["category"] = test_frame["category"].replace({0 : 'cat', 1: 'dog'})

total_train = train_frame.shape[0]
total_test = test_frame.shape[0]
batch_size = 15

#train generator, 이미지를 회전 혹은 줌, 상하/좌우 반전을 통해 여러개생성
train_datagen = ImageDataGenerator(
    rotation_range = 15,
    rescale = 1./255,
    shear_range = 0.1,
    zoom_range = 0.2,
    horizontal_flip = True,
    width_shift_range = 0.1,
    height_shift_range = 0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_frame,
    "./data/training_set/",
    x_col = "filename",
    y_col = "category",
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

#validation gen
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    test_frame,
    "./data/test_set/",
    x_col = 'filename',
    y_col = 'category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

#fit_generator는 generator를 사용해서 만든 것을 돌릴 수 있다. 일반 fit에 비해 많은 데이터를 빠르게 돌려볼 수 있음
epochs = 3 if FAST_RUN else 50
history = model.fit_generator(
    train_generator,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = total_test//batch_size,
    steps_per_epoch = total_train//batch_size,
    callbacks=callbacks
)

model.save_weights("model.h5")
