import numpy as np
import tensorflow as tf
#from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from tensorflow.python.keras.constraints import maxnorm
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt


import keras
#fashion_mnist = tf.python.keras.datasets.fashion_mnist
print("TF Version", tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print (" load data")
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()


train_images = train_images / 255.0
test_images = test_images / 255.0

print (train_images.shape, train_labels, len(train_labels))

# ---- model

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=3)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# --- use model ----

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

print (predictions[0])
predicted = np.argmax(predictions[0])

print ("best "+str(class_names[predicted])+" "+str(predicted))

plt.figure()
plt.imshow(test_images[predicted])
plt.colorbar()
plt.grid(False)
plt.show()