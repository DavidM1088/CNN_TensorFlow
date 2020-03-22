#https://www.tensorflow.org/tutorials/load_data/images
import numpy as np
import os, os.path
import tensorflow as tf
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras

import matplotlib.pyplot as plt

root_dir = "./images"
image_count = 4
fast = False
slow = False
cifar_10 = True

def show_batch(image_batch, label_batch):
    #print ('Labels:', label_batch)
    plt.figure(figsize=(10,10))
    class_names = [dir[0] for dir in os.walk(root_dir + '/train') if len(dir[1]) == 0]
    class_names = ['aussie','basset','samoyed']
    for n in range(20):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        #titl = class_names[label_batch[n]==1][0].title()
        ii = int(label_batch[n])
        titl = class_names[ii]
        plt.title(titl)
        plt.axis('off')
    plt.show()

total_training_images = 0
total_validate_images = 0
total_test_images = 0

for c in range(0,3):
    if c==0: f_dir = '/train'
    if c==1: f_dir = '/test'
    if c==2: f_dir = '/validate'
    cat_names = [dir[0] for dir in os.walk(root_dir + f_dir) if len(dir[1]) == 0]
    for cat_dir in cat_names:
        onlyfiles = next(os.walk(cat_dir))[2]
        if c==0: total_training_images += len(onlyfiles)
        if c==1: total_test_images += len(onlyfiles)
        if c==2: total_validate_images += len(onlyfiles)

# One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE.
# Since one epoch is too big to feed to the computer at once we divide it in several smaller batches.
# batch : you divide dataset into Number of Batches or sets or parts.
# batch size: Total number of training examples present in a single batch.
# Iterations is the number of batches needed to complete one epoch.

EPOCHS = 20 #15

STEPS_PER_EPOCH = 2 # how many batches of training data per epoch
TRAINING_BATCH_SIZE = total_training_images // STEPS_PER_EPOCH
VALIDATE_BATCH_SIZE = total_validate_images // STEPS_PER_EPOCH

IMG_HEIGHT = 64
IMG_WIDTH = 64

print ("STEPS:" + str(STEPS_PER_EPOCH) + " BATCH_SIZE:" + str(TRAINING_BATCH_SIZE) + " total_training:" + str(total_training_images) + ' epochs:' + str(EPOCHS))
print ("STEPS:" + str(STEPS_PER_EPOCH) + " BATCH_SIZE:" + str(VALIDATE_BATCH_SIZE) + " total_validate:" + str(total_validate_images) + ' epochs:' + str(EPOCHS))

# ---- image generators -----

if cifar_10:

    train_image_gen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # https://keras.io/examples/cifar10_cnn/
    from keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    num_classes = 10
    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    input_shape = x_train.shape[1:]
    print(input_shape)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    #train_data_gen = ImageDataGenerator.flow(x_train, y_train, batch_size=TRAINING_BATCH_SIZE)
    #validation_image_gen = ImageDataGenerator(rescale=1. / 255)  # dont apply image augmentation to validation
    #test_image_gen = ImageDataGenerator(rescale=1. / 255)
else:
    train_image_gen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=45,
        width_shift_range=.15,
        height_shift_range=.15,
        horizontal_flip=True,
        zoom_range=0.5
    )

    train_image_gen = ImageDataGenerator(rescale=1. / 255)\

    validation_image_gen = ImageDataGenerator(rescale=1. / 255) # dont apply image augmentation to validation
    test_image_gen = ImageDataGenerator(rescale=1. / 255)

    # ----- data gens -----

    train_data_gen = train_image_gen.flow_from_directory(batch_size=TRAINING_BATCH_SIZE, directory=root_dir+'/train',shuffle=True,target_size=(IMG_HEIGHT, IMG_WIDTH),class_mode='binary')

    val_data_gen = validation_image_gen.flow_from_directory(batch_size=VALIDATE_BATCH_SIZE,directory=root_dir+'/validate',target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='binary')

    test_data_gen = validation_image_gen.flow_from_directory(batch_size=VALIDATE_BATCH_SIZE, directory=root_dir+'/test', target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='binary')

    #image_batch, label_batch = next(train_image_gen)
    image_batch, label_batch = test_data_gen.next()
    print (label_batch)
    #print (image_batch)

# show_batch(image_batch, label_batch)

#see how apply rotate, flip etc work on original images
#augmented_images = [train_data_gen[0][0][0] for i in range(15)]
#plotImages(augmented_images)

# ===================== build model
input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)

if False and cifar_10:
    IMG_HEIGHT = 28
    IMG_WIDTH = 28
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

if cifar_10:
    IMG_WIDTH = input_shape=x_train.shape[1:][0]
    IMG_HEIGHT = input_shape = x_train.shape[1:][1]
    input_shape  = x_train.shape[1:]
    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=input_shape),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1)
    ])
    # ====================
    from keras.layers import Dense, Dropout, Activation, Flatten
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
print (model.summary())

# =================== train model

if slow :
    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        validation_data=val_data_gen,
        validation_steps = STEPS_PER_EPOCH)

if cifar_10:
    history = model.fit(x_train, y_train,
              batch_size=STEPS_PER_EPOCH,
              epochs=EPOCHS,
              validation_data=(x_test, y_test),
              shuffle=True)

if False:
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss=history.history['loss']
    val_loss=history.history['val_loss']

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

#scores = model.evaluate(X_test, y_test, verbose=0)
#print("Accuracy: %.2f%%" % (scores[1]*100))

#predictions = model()

print('validate eval:'+str(model.evaluate(val_data_gen)))
print('test eval:'+str(model.evaluate(test_data_gen)))

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

# x = test_data_gen.next()
# y = probability_model(x[:5])
# print (y)
