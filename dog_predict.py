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
from  PIL import Image
import random

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_dogbreed_trained_model.h5'

# Save model and weights
model_path = os.path.join(save_dir, model_name)
model = keras.models.load_model(model_path)

cat_file = save_dir+'/category_labels.txt'
category_labels = [line.strip() for line in open(cat_file, 'r')]

# ================================== predict

def convert_to_array(img, img_size):
    im = cv2.imread(img)
    img = Image.fromarray(im, 'RGB')
    image = img.resize((img_size, img_size))
    return np.array(image)

def get_animal_name(label):
    if label==0:
        return "cat"
    if label==1:
        return "dog"
    if label==2:
        return "bird"
    if label==3:
        return "fish"

total_correct = 0
total_predicts = 0

def predict_breed(file, img_size):
    global  total_predicts, total_correct

    im = cv2.imread(file)  # make sure same as model training
    im = cv2.resize(im, (img_size, img_size))
    x_data = []
    x_data.append(im)
    x_data = np.array(x_data)

    score = model.predict(x_data, verbose=0)

    label_index = np.argmax(score)
    pred_cat = category_labels[label_index]

    segs = file.split('/')
    actual_cat = segs[len(segs)-2]
    wrong = ""
    if pred_cat == actual_cat:
        total_correct += 1
    else:
        wrong = "===>WRONG"
    total_predicts += 1
    perc = (total_correct*100)/total_predicts

    print(wrong+" predicted breed:"+pred_cat + ' actual:'+actual_cat + '\t\t\ttotal_correct:' + str(total_correct) + ' total predicts:' + str(total_predicts) + ' perc:'+ str(perc))

predict = []

def show(img_file, img_size):
    img = cv2.imread(img_file)  # make sure same as model training
    #img = cv2.resize(img, (img_size, img_size))
    #cv2.imwrite('color_img.jpg', im)
    #cv2.imshow("image", im);
    #cv2.waitKey();
    # img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.show()


for root, dirs, files in os.walk("images/dogs"):
    for name in files:
        path = os.path.join(root, name)
        if path.find('.jpg') < 0:
            continue
        predict.append(path)

random.shuffle(predict)
for image_file in predict:
    predict_breed(image_file, 64)
    s = input("show image ?")
    if s.lower() == 'y':
        show(image_file, 64)

