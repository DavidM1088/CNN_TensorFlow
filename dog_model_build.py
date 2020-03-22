from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

do_cifar = False
batch_size = 64
epochs = 50  #100
data_augmentation = False
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_dogbreed_trained_model.h5'

def save_model(model, category_models):
    # serialize model to json
    save_dir = './saved_models'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # === Json save
    json_model = model.to_json()
    # save the model architecture to JSON file
    with open(save_dir+'/saved_model.json', 'w') as json_file:
        json_file.write(json_model)
    # saving the weights of the model
    model.save_weights(save_dir+'/saved_model_weights.json')
    # Model loss and accuracy
    #loss, acc = model.evaluate(test_images, test_labels, verbose=2)

    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    #model.save('path_to_saved_model',                          save_format='tf')
    #new_model = keras.models.load_model('path_to_my_model.h5')
    # Check that the state is preserved
    #new_predictions = new_model.predict(x_test)
    #np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)

    # Note that the optimizer state is preserved as well:
    # you can resume training where you left off.

    with open(save_dir+"/category_labels.txt", "w") as output:
        for c in category_labels:
            output.write(str(c)+"\n")

def show_batch(image_batch, label_batch):
    fig = plt.figure(figsize=(12, 12))
    rows = cols = 5
    n = min(rows * cols, len(image_batch))
    axes = []
    for n in range(n):
        axes.append(fig.add_subplot(rows, cols, n + 1))
        axes[-1].set_title(":" + str(label_batch[n]))
        axes[-1].set_xticks([])
        axes[-1].set_yticks([])
        plt.imshow(image_batch[n]) #, alpha=0.25)
    plt.show()

def get_data():
    if do_cifar:
        img_size = 32
        # The data, split between train and test sets:
        (x_data, y_train), (x_test, y_test) = cifar10.load_data()
        print('x_train shape:', x_data.shape)
        print(x_data.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
        #show_batch(x_train, y_train)
        return (x_data, y_train), (x_test, y_test), 10
    else:
        img_size = 64
        root_dir = "./images/dogs"
        x_data = []
        y_data = []
        category_labels = []
        image_cnt = 0

        for typ in os.scandir(root_dir):
            if not os.path.isdir(typ.path):
                continue
            for cat in os.scandir(typ.path):
                if not os.path.isdir(cat.path):
                    continue
                path = str(cat.path)
                last = path.rindex('/')
                scat = path[last+1 : len(path)]
                if not scat in category_labels:
                    category_labels.append(scat)
                for f in os.scandir(cat.path):
                    if f.path.find('.jpg') > 0:
                        #print ("adding", f.path)
                        im = cv2.imread(f.path) #, cv2.IMREAD_COLOR)
                        im = cv2.resize(im, (img_size, img_size))
                        x_data.append(im)
                        y_data.append(category_labels.index(scat))
                        image_cnt += 1
                        if (image_cnt % 500) == 0:
                            print ('images loaded from files:'+str(image_cnt))

    # See - https://mc.ai/a-simple-animal-classifier-from-scratch-using-keras/
    x_data = np.array(x_data)
    num_classes = len(category_labels)
    if False:
        s = np.arange(x_data.shape[0])
        np.random.shuffle(s)
        x_data = x_data[s]
        y_data = y_data[s]

    # Take 90% of data in train set and 10% in test set
    data_length = len(x_data)
    (x_train, x_test) = x_data[(int)(0.1 * data_length):], x_data[:(int)(0.1 * data_length)]
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    (y_train, y_test) = y_data[(int)(0.1 * data_length):], y_data[:(int)(0.1 * data_length)]
    print ('total training images:'+str(data_length)+', training:'+str(len(x_train))+', test:'+str(len(x_test)))
    return (x_train, y_train), (x_test, y_test), num_classes, img_size, category_labels

(x_train, y_train), (x_test, y_test), num_classes, img_size, category_labels = get_data()

#show_batch(x_train, y_train)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
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

# initiate RMSprop optimizer
opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#show_batch(x_train, y_train)
#x_train /= 255 TODO WHY????
#x_test /= 255
#show_batch(x_train, y_train)

if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
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

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        epochs=epochs, validation_data=(x_test, y_test), workers=4,
                        callbacks=[tensorboard_callback])

# Save model and weights
save_model(model, category_labels)

if True:
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

# Score trained model.
print ('\nAccuracy: tot_train:'+str(len(x_train))+", tot test:"+str(len(x_test)))
train_scores = model.evaluate(x_train, y_train, verbose=0)
test_scores = model.evaluate(x_test, y_test, verbose=0)
print('Train loss:', train_scores[0],  'Train accuracy:', train_scores[1])
print('Test loss:', test_scores[0],  'Test accuracy:', test_scores[1])

#print (model.metrics_names)


