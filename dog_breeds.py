#https://www.tensorflow.org/tutorials/load_data/images
import numpy as np
import os, os.path
import tensorflow as tf
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

root_dir = "./images"
image_count = 4
fast = False

def show_batch(image_batch, label_batch):
    #print ('Labels:', label_batch)
    plt.figure(figsize=(10,10))
    for n in range(5):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        #plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
        plt.axis('off')
    plt.show()


total_training_files = 0
total_validate_files = 0
total_test_files = 0

for c in range(0,3):
    if c==0: f_dir = '/train'
    if c==1: f_dir = '/test'
    if c==2: f_dir = '/validate'
    cat_names = [dir[0] for dir in os.walk(root_dir + f_dir) if len(dir[1]) == 0]
    for cat_dir in cat_names:
        onlyfiles = next(os.walk(cat_dir))[2]
        if c==0: total_training_files += len(onlyfiles)
        if c==1: total_test_files += len(onlyfiles)
        if c==2: total_validate_files += len(onlyfiles)

# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     width_shift_range=0.15,
#     height_shift_range=0.15,
#     shear_range=0.15,
#     zoom_range=0.15,
#     horizontal_flip=True,
#     fill_mode='nearest')

#image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
# train_generator = image_generator.flow_from_directory(
#     directory=r"./images/train/",
#     target_size=(224, 224),
#     color_mode="rgb",
#     batch_size=32,
#     class_mode="categorical",
#     shuffle=True,
#     seed=42
# )

print (root_dir, os.path.isdir(root_dir))

# train_data_gen_slow = image_generator.flow_from_directory(directory=str(training_dir/train),
#                                                      batch_size=BATCH_SIZE,
#                                                      shuffle=True,
#                                                      target_size=(IMG_HEIGHT, IMG_WIDTH))
#                                                      #classes = CLASS_NAMES) # DONT USE THIS

#image_batch, label_batch = next(train_data_gen)
#show_batch(image_batch, label_batch)

# One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE.
# Since one epoch is too big to feed to the computer at once we divide it in several smaller batches.
# batch : you divide dataset into Number of Batches or sets or parts.
# batch size: Total number of training examples present in a single batch.
# Iterations is the number of batches needed to complete one epoch.

EPOCHS = 30 #15

STEPS_PER_EPOCH = 2 # how many batches of training data per epoch
TRAINING_BATCH_SIZE = total_training_files // STEPS_PER_EPOCH
VALIDATE_BATCH_SIZE = total_validate_files // STEPS_PER_EPOCH

IMG_HEIGHT = 150
IMG_WIDTH = 150
print ("STEPS:" + str(STEPS_PER_EPOCH) + " BATCH_SIZE:" + str(TRAINING_BATCH_SIZE) + " total_training:" + str(total_training_files) + ' epochs:' + str(EPOCHS))
print ("STEPS:" + str(STEPS_PER_EPOCH) + " BATCH_SIZE:" + str(VALIDATE_BATCH_SIZE) + " total_validate:" + str(total_validate_files) + ' epochs:' + str(EPOCHS))

# ==================== Faster
if fast:
    import pathlib

    train_dir = pathlib.Path(root_dir+'/train')
    test_dir = pathlib.Path(root_dir+'/test')
    validate_dir = pathlib.Path(root_dir+'/validate')

    training_data_set = tf.data.Dataset.list_files(str(train_dir / '*/*'))
    test_data_set = tf.data.Dataset.list_files(str(test_dir / '*/*'))
    validation_data_set = tf.data.Dataset.list_files(str(validate_dir / '*/*'))

    for f in training_data_set.take(5):
      print(f.numpy())

    def get_label(file_path):
      # convert the path to a list of path components
      parts = tf.strings.split(file_path, os.path.sep)
      # The second to last is the class-directory
      return parts[-2] == CLASS_NAMES

    def decode_img(img):
      # convert the compressed string to a 3D uint8 tensor
      img = tf.image.decode_jpeg(img, channels=3)
      # Use `convert_image_dtype` to convert to floats in the [0,1] range.
      img = tf.image.convert_image_dtype(img, tf.float32)
      # resize the image to the desired size.
      return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

    def process_path(file_path):
      label = get_label(file_path)
      # load the raw data from the file as a string
      img = tf.io.read_file(file_path)
      img = decode_img(img)
      return img, label

    training_labeled_ds = training_data_set.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_labeled_ds = test_data_set.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    validate_labeled_ds = validation_data_set.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
        # This is a small dataset, only load it once, and keep it in memory.
        # use `.cache(filename)` to cache preprocessing work for datasets that don't
        # fit in memory.
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()

        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
        # Repeat forever
        ds = ds.repeat()
        ds = ds.batch(BATCH_SIZE)
        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

    default_timeit_steps = 1000

    def timeit(ds, steps=default_timeit_steps):
        start = time.time()
        it = iter(ds)
        for i in range(steps):
            batch = next(it)
            if i % 10 == 0:
                print('.', end='')
        print()
        end = time.time()

        duration = end - start
        print("{} batches: {} s".format(steps, duration))
        print("{:0.5f} Images/s".format(BATCH_SIZE * steps / duration))


    prepared_data_set_train = prepare_for_training(training_labeled_ds)
    prepared_data_set_validate = prepare_for_training(validate_labeled_ds)

    # sample_training_images, _ = next(prepared_data_set)
    # image_batch, label_batch = next(iter(prepared_data_set_train))

    # timeit(train_data_gen_slow)
    # timeit(prepared_data_set_train)
    # timeit(prepared_data_set_validate)
    prepared_data_set_train
    prepared_data_set_validate

else:
    def plotImages(images_arr):
        fig, axes = plt.subplots(1, 5, figsize=(20, 20))
        axes = axes.flatten()
        for img, ax in zip(images_arr, axes):
            ax.imshow(img)
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    #train_image_gen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)
    train_image_gen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=45,
        width_shift_range=.15,
        height_shift_range=.15,
        horizontal_flip=True,
        zoom_range=0.5
    )

    validation_image_gen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)
    # validation_image_gen = ImageDataGenerator( # dont apply to validation
    #     rescale=1. / 255,
    #     rotation_range=45,
    #     width_shift_range=.15,
    #     height_shift_range=.15,
    #     horizontal_flip=True,
    #     zoom_range=0.5
    # )

    train_data_gen = train_image_gen.flow_from_directory(batch_size=TRAINING_BATCH_SIZE,
                                                               directory=root_dir+'/train',
                                                               shuffle=True,
                                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                               class_mode='binary')
    # cnt = 0
    # for x in train_data_gen.filenames:
    #     #print(x)
    #     cnt += 1
    # print("files in generator:"+str(cnt))

    val_data_gen = validation_image_gen.flow_from_directory(batch_size=VALIDATE_BATCH_SIZE,
                                                                  directory=root_dir+'/validate',
                                                                  target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                  class_mode='binary')

    # see how apply rotate, flip etc work on original images
    # augmented_images = [train_data_gen[0][0][0] for i in range(5)]
    # plotImages(augmented_images)


# for image, label in training_labeled_ds.take(1):
#   print("Image shape: ", image.numpy().shape)
#   print("Label: ", label.numpy())

# ===================== build model

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
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

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
print (model.summary())

num_categories  = len(os.listdir(root_dir+'/train'))
print ("category count:", num_categories)

# =================== train model

if fast:
    history = model.fit_generator(
        prepared_data_set_train,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        validation_data=prepared_data_set_validate,
        validation_steps= STEPS_PER_EPOCH)
else:
    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        validation_data=val_data_gen,
        validation_steps = STEPS_PER_EPOCH)


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