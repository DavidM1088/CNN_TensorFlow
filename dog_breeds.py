#https://www.tensorflow.org/tutorials/load_data/images
import numpy as np
import os
import tensorflow as tf
import time

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from tensorflow.python.keras.constraints import maxnorm
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from scraper import Scraper

training_dir = "./images/train/"

image_count = 4

def show_batch(image_batch, label_batch):
    #print ('Labels:', label_batch)
    plt.figure(figsize=(10,10))
    for n in range(5):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        #plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
        plt.axis('off')
    plt.show()

CLASS_NAMES = [dir[0] for dir in os.walk(training_dir) if len(dir[1]) == 0]
print (CLASS_NAMES)
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     width_shift_range=0.15,
#     height_shift_range=0.15,
#     shear_range=0.15,
#     zoom_range=0.15,
#     horizontal_flip=True,
#     fill_mode='nearest')

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
# train_generator = image_generator.flow_from_directory(
#     directory=r"./images/train/",
#     target_size=(224, 224),
#     color_mode="rgb",
#     batch_size=32,
#     class_mode="categorical",
#     shuffle=True,
#     seed=42
# )

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)
print (training_dir, os.path.isdir(training_dir))
# train_data_gen_slow = image_generator.flow_from_directory(directory=str(training_dir),
#                                                      batch_size=BATCH_SIZE,
#                                                      shuffle=True,
#                                                      target_size=(IMG_HEIGHT, IMG_WIDTH))
#                                                      #classes = CLASS_NAMES) # DONT USE THIS

#image_batch, label_batch = next(train_data_gen)
#show_batch(image_batch, label_batch)

# ==================== Faster
import pathlib

data_dir = pathlib.Path(training_dir)
training_data_set = tf.data.Dataset.list_files(str(data_dir / '*/*'))

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

for image, label in training_labeled_ds.take(1):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())

# =================

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
    if i%10 == 0:
      print('.',end='')
  print()
  end = time.time()

  duration = end-start
  print("{} batches: {} s".format(steps, duration))
  print("{:0.5f} Images/s".format(BATCH_SIZE*steps/duration))

prepared_data_set = prepare_for_training(training_labeled_ds)
#filecache_ds = prepare_for_training(labeled_ds, cache="./flowers.tfcache")timeit(filecache_ds)

#sample_training_images, _ = next(prepared_data_set)
image_batch, label_batch = next(iter(prepared_data_set))

#timeit(train_data_gen_slow)
timeit(prepared_data_set)

# ===================== build model

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
print (model.summary())

batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

num_categories  = len(os.listdir(training_dir))
print ("category count:", num_categories)
total_train = 10

# =================== train model
history = model.fit_generator(
    prepared_data_set,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

if False:
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss=history.history['loss']
    val_loss=history.history['val_loss']

    epochs_range = range(epochs)

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