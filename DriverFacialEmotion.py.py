# -*- coding: utf-8 -*-
"""Driver facial emoton detection
"""

!pip install tensorflow==1.14

from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
tf.__version__

"""# facial emoton detection

## Mapping real-world to ML Problem
"""

import os,shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import glob
import cv2
from sklearn.model_selection import train_test_split
from keras.layers import Dropout, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, load_model
from keras.applications import VGG16
from sklearn.metrics import accuracy_score, confusion_matrix

"""#################################################################################################################################"""

shutil.copytree('/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Human_org', '/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Human')

shutil.copytree('/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Animated_org', '/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Animated')

"""##################################################################################################################

## 1. Reading the Data of Human Images

### Angry
"""

human_angry = glob.glob("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Human/Angry/*")
#human_angry.remove('/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Human/Angry\\Thumbs.db')
print("Number of images in Angry emotion = "+str(len(human_angry)))

human_angry_folderName = [os.path.dirname(x) for x in human_angry]
human_angry_imageName = [os.path.basename(x) for x in human_angry]
human_angry_emotion = [["Angry"]*len(human_angry)][0]
human_angry_label = [1]*len(human_angry)

len(human_angry_folderName), len(human_angry_imageName), len(human_angry_emotion), len(human_angry_label)

df_angry = pd.DataFrame()
df_angry["folderName"] = human_angry_folderName
df_angry["imageName"] = human_angry_imageName
df_angry["Emotion"] = human_angry_emotion
df_angry["Labels"] = human_angry_label
df_angry.head()

"""### Disgust"""

human_disgust = glob.glob("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Human/Disgust/*")
#human_disgust.remove('../Data/Human/Disgust\\Thumbs.db')
print("Number of images in Disgust emotion = "+str(len(human_disgust)))

human_disgust_folderName = [os.path.dirname(x) for x in human_disgust]
human_disgust_imageName = [os.path.basename(x) for x in human_disgust]
human_disgust_emotion = [["Disgust"]*len(human_disgust)][0]
human_disgust_label = [2]*len(human_disgust)

len(human_disgust_folderName), len(human_disgust_imageName), len(human_disgust_emotion), len(human_disgust_label)

df_disgust = pd.DataFrame()
df_disgust["folderName"] = human_disgust_folderName
df_disgust["imageName"] = human_disgust_imageName
df_disgust["Emotion"] = human_disgust_emotion
df_disgust["Labels"] = human_disgust_label
df_disgust.head()

"""### Fear"""

human_fear = glob.glob("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Human/Fear/*")
#human_fear.remove('../Data/Human/Fear\\Thumbs.db')
print("Number of images in Fear emotion = "+str(len(human_fear)))

human_fear_folderName = [os.path.dirname(x) for x in human_fear]
human_fear_imageName = [os.path.basename(x) for x in human_fear]
human_fear_emotion = [["Fear"]*len(human_fear)][0]
human_fear_label = [3]*len(human_fear)

len(human_fear_folderName), len(human_fear_imageName), len(human_fear_emotion), len(human_fear_label)

df_fear = pd.DataFrame()
df_fear["folderName"] = human_fear_folderName
df_fear["imageName"] = human_fear_imageName
df_fear["Emotion"] = human_fear_emotion
df_fear["Labels"] = human_fear_label
df_fear.head()

"""### Happy"""

human_happy = glob.glob("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Human/Happy/*")
#human_happy.remove('../Data/Human/Happy\\Thumbs.db')
print("Number of images in Happy emotion = "+str(len(human_happy)))

human_happy_folderName = [os.path.dirname(x) for x in human_happy]
human_happy_imageName = [os.path.basename(x) for x in human_happy]
human_happy_emotion = [["Happy"]*len(human_happy)][0]
human_happy_label = [4]*len(human_happy)

len(human_happy_folderName), len(human_happy_imageName), len(human_happy_emotion), len(human_happy_label)

df_happy = pd.DataFrame()
df_happy["folderName"] = human_happy_folderName
df_happy["imageName"] = human_happy_imageName
df_happy["Emotion"] = human_happy_emotion
df_happy["Labels"] = human_happy_label
df_happy.head()



"""### Neutral"""

human_neutral = glob.glob("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Human/Neutral/*")
#human_neutral.remove('../Data/Human/Neutral\\Thumbs.db')
print("Number of images in Neutral emotion = "+str(len(human_neutral)))

human_neutral_folderName = [os.path.dirname(x) for x in human_neutral]
human_neutral_imageName = [os.path.basename(x) for x in human_neutral]
human_neutral_emotion = [["Neutral"]*len(human_neutral)][0]
human_neutral_label = [5]*len(human_neutral)

len(human_neutral_folderName), len(human_neutral_imageName), len(human_neutral_emotion), len(human_neutral_label)

df_neutral = pd.DataFrame()
df_neutral["folderName"] = human_neutral_folderName
df_neutral["imageName"] = human_neutral_imageName
df_neutral["Emotion"] = human_neutral_emotion
df_neutral["Labels"] = human_neutral_label
df_neutral.head()

"""### Sad"""

human_sad = glob.glob("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Human/Sad/*")
#human_sad.remove('../Data/Human/Sad\\Thumbs.db')
print("Number of images in Sad emotion = "+str(len(human_sad)))

human_sad_folderName = [os.path.dirname(x) for x in human_sad]
human_sad_imageName = [os.path.basename(x) for x in human_sad]
human_sad_emotion = [["Sad"]*len(human_sad)][0]
human_sad_label = [6]*len(human_sad)

len(human_sad_folderName), len(human_sad_imageName), len(human_sad_emotion), len(human_sad_label)

df_sad = pd.DataFrame()
df_sad["folderName"] = human_sad_folderName
df_sad["imageName"] = human_sad_imageName
df_sad["Emotion"] = human_sad_emotion
df_sad["Labels"] = human_sad_label
df_sad.head()

"""### Surprise"""

human_surprise = glob.glob("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Human/Surprice/*")
#human_surprise.remove('../Data/Human/Surprise\\Thumbs.db')
print("Number of images in Surprise emotion = "+str(len(human_surprise)))

human_surprise_folderName = [os.path.dirname(x) for x in human_surprise]
human_surprise_imageName = [os.path.basename(x) for x in human_surprise]
human_surprise_emotion = [["Surprise"]*len(human_surprise)][0]
human_surprise_label = [7]*len(human_surprise)

len(human_surprise_folderName), len(human_surprise_imageName), len(human_surprise_emotion), len(human_surprise_label)

df_surprise = pd.DataFrame()
df_surprise["folderName"] = human_surprise_folderName
df_surprise["imageName"] = human_surprise_imageName
df_surprise["Emotion"] = human_surprise_emotion
df_surprise["Labels"] = human_surprise_label
df_surprise.head()

length = df_angry.shape[0] + df_disgust.shape[0] + df_fear.shape[0] + df_happy.shape[0] + df_neutral.shape[0] + df_sad.shape[0] + df_surprise.shape[0]
print("Total number of images in all the emotions = "+str(length))

"""### Concatenating all dataframes"""

frames = [df_angry, df_disgust, df_fear, df_happy, df_neutral, df_sad, df_surprise]
Final_human = pd.concat(frames)
Final_human.shape

Final_human.reset_index(inplace = True, drop = True)
Final_human = Final_human.sample(frac = 1.0)   #shuffling the dataframe
Final_human.reset_index(inplace = True, drop = True)
Final_human.head()

"""## 2. Train, CV and Test Split for Human Images"""

df_human_train_data, df_human_test = train_test_split(Final_human, stratify=Final_human["Labels"], test_size = 0.197860)
df_human_train, df_human_cv = train_test_split(df_human_train_data, stratify=df_human_train_data["Labels"], test_size = 0.166666)
df_human_train.shape, df_human_cv.shape, df_human_test.shape

df_human_train.reset_index(inplace = True, drop = True)
df_human_train.to_pickle("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Dataframes/Human/df_human_train.pkl")

df_human_cv.reset_index(inplace = True, drop = True)
df_human_cv.to_pickle("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Dataframes/Human/df_human_cv.pkl")

df_human_test.reset_index(inplace = True, drop = True)
df_human_test.to_pickle("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Dataframes/Human/df_human_test.pkl")

df_human_train = pd.read_pickle("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Dataframes/Human/df_human_train.pkl")
df_human_train.head()

df_human_train.shape

df_human_cv = pd.read_pickle("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Dataframes/Human/df_human_cv.pkl")
df_human_cv.head()

df_human_cv.shape

df_human_test = pd.read_pickle("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Dataframes/Human/df_human_test.pkl")
df_human_test.head()

df_human_test.shape

"""## 3. Analysing Data of Human Images
### Distribution of class labels in Train, CV and Test
"""

df_temp_train = df_human_train.sort_values(by = "Labels", inplace = False)
df_temp_cv = df_human_cv.sort_values(by = "Labels", inplace = False)
df_temp_test = df_human_test.sort_values(by = "Labels", inplace = False)

TrainData_distribution = df_human_train["Emotion"].value_counts().sort_index()
CVData_distribution = df_human_cv["Emotion"].value_counts().sort_index()
TestData_distribution = df_human_test["Emotion"].value_counts().sort_index()

TrainData_distribution_sorted = sorted(TrainData_distribution.items(), key = lambda d: d[1], reverse = True)
CVData_distribution_sorted = sorted(CVData_distribution.items(), key = lambda d: d[1], reverse = True)
TestData_distribution_sorted = sorted(TestData_distribution.items(), key = lambda d: d[1], reverse = True)

fig = plt.figure(figsize = (10, 6))
ax = fig.add_axes([0,0,1,1])
ax.set_title("Count of each Emotion in Train Data", fontsize = 20)
sns.countplot(x = "Emotion", data = df_temp_train)
plt.grid()
for i in ax.patches:
    ax.text(x = i.get_x() + 0.2, y = i.get_height()+1.5, s = str(i.get_height()), fontsize = 20, color = "grey")
plt.xlabel("")
plt.ylabel("Count", fontsize = 15)
plt.tick_params(labelsize = 15)
plt.xticks(rotation = 40)
plt.show()

for i in TrainData_distribution_sorted:
    print("Number of training data points in class "+str(i[0])+" = "+str(i[1])+ "("+str(np.round(((i[1]/df_temp_train.shape[0])*100), 4))+"%)")

print("-"*80)

fig = plt.figure(figsize = (10, 6))
ax = fig.add_axes([0,0,1,1])
ax.set_title("Count of each Emotion in Validation Data", fontsize = 20)
sns.countplot(x = "Emotion", data = df_temp_cv)
plt.grid()
for i in ax.patches:
    ax.text(x = i.get_x() + 0.27, y = i.get_height()+0.2, s = str(i.get_height()), fontsize = 20, color = "grey")
plt.xlabel("")
plt.ylabel("Count", fontsize = 15)
plt.tick_params(labelsize = 15)
plt.xticks(rotation = 40)
plt.show()

for i in CVData_distribution_sorted:
    print("Number of training data points in class "+str(i[0])+" = "+str(i[1])+ "("+str(np.round(((i[1]/df_temp_cv.shape[0])*100), 4))+"%)")

print("-"*80)

fig = plt.figure(figsize = (10, 6))
ax = fig.add_axes([0,0,1,1])
ax.set_title("Count of each Emotion in Test Data", fontsize = 20)
sns.countplot(x = "Emotion", data = df_temp_test)
plt.grid()
for i in ax.patches:
    ax.text(x = i.get_x() + 0.27, y = i.get_height()+0.2, s = str(i.get_height()), fontsize = 20, color = "grey")
plt.xlabel("")
plt.ylabel("Count", fontsize = 15)
plt.tick_params(labelsize = 15)
plt.xticks(rotation = 40)
plt.show()

for i in TestData_distribution_sorted:
    print("Number of training data points in class "+str(i[0])+" = "+str(i[1])+ "("+str(np.round(((i[1]/df_temp_test.shape[0])*100), 4))+"%)")

"""## 4. Pre-Processing Human Images

### 4.1 Converting all the images to grayscale and save them
"""

def convt_to_gray(df):
    count = 0
    for i in range(len(df)):
        path1 = df["folderName"][i]
        path2 = df["imageName"][i]
        img = cv2.imread(os.path.join(path1, path2))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(path1, path2), gray)
        count += 1
    print("Total number of images converted and saved = "+str(count))

convt_to_gray(df_human_train)

convt_to_gray(df_human_cv)

convt_to_gray(df_human_test)

"""### 4.2 Detecting face in image using HAAR then crop it then resize then save the image"""

#detect the face in image using HAAR cascade then crop it then resize it and finally save it.
face_cascade = cv2.CascadeClassifier('/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/haarcascade_frontalface_default.xml') 
#download this xml file from link: https://github.com/opencv/opencv/tree/master/data/haarcascades.
def face_det_crop_resize(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        face_clip = img[y:y+h, x:x+w]  #cropping the face in image
        cv2.imwrite(img_path, cv2.resize(face_clip, (350, 350)))  #resizing image then saving it

for i, d in df_human_train.iterrows():
    img_path = os.path.join(d["folderName"], d["imageName"])
    face_det_crop_resize(img_path)

for i, d in df_human_cv.iterrows():
    img_path = os.path.join(d["folderName"], d["imageName"])
    face_det_crop_resize(img_path)

for i, d in df_human_test.iterrows():
    img_path = os.path.join(d["folderName"], d["imageName"])
    face_det_crop_resize(img_path)

"""## 5. Reading the Data of Animated Images

### Angry
"""

anime_angry = glob.glob("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Animated/Angry/*.png")
print("Number of images in Angry emotion = "+str(len(anime_angry)))

anime_angry_folderName = [os.path.dirname(x) for x in anime_angry]
anime_angry_imageName = [os.path.basename(x) for x in anime_angry]
anime_angry_emotion = [["Angry"]*len(anime_angry)][0]
anime_angry_label = [1]*len(anime_angry)

len(anime_angry_folderName), len(anime_angry_imageName), len(anime_angry_emotion), len(anime_angry_label)

df_angry = pd.DataFrame()
df_angry["folderName"] = anime_angry_folderName
df_angry["imageName"] = anime_angry_imageName
df_angry["Emotion"] = anime_angry_emotion
df_angry["Labels"] = anime_angry_label
df_angry.head()

df_angry = df_angry.sample(frac = 1.0) #shuffling dataframe
df_angry_reduced = df_angry.sample(n = 230)  #taking only 1300 random images
df_angry_reduced.shape

#removing all the extra images from storage
df_angry_reducedIndx = df_angry_reduced.index
count = 0
for i, d in df_angry.iterrows():
    if i not in df_angry_reducedIndx:
        os.remove(os.path.join(d["folderName"], d["imageName"]))
        count += 1
print("Total number of images removed = "+str(count))

"""### Disgust"""

anime_disgust = glob.glob("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Animated/Disgust/*.png")
print("Number of images in Disgust emotion = "+str(len(anime_disgust)))

anime_disgust_folderName = [os.path.dirname(x) for x in anime_disgust]
anime_disgust_imageName = [os.path.basename(x) for x in anime_disgust]
anime_disgust_emotion = [["Disgust"]*len(anime_disgust)][0]
anime_disgust_label = [2]*len(anime_disgust)

len(anime_disgust_folderName), len(anime_disgust_imageName), len(anime_disgust_emotion), len(anime_disgust_label)

df_disgust = pd.DataFrame()
df_disgust["folderName"] = anime_disgust_folderName
df_disgust["imageName"] = anime_disgust_imageName
df_disgust["Emotion"] = anime_disgust_emotion
df_disgust["Labels"] = anime_disgust_label
df_disgust.head()

df_disgust = df_disgust.sample(frac = 1.0) #shuffling dataframe
df_disgust_reduced = df_disgust.sample(n = 240)  #taking only 1300 random images
df_disgust_reduced.shape

#removing all the extra images from storage
df_disgust_reducedIndx = df_disgust_reduced.index
count = 0
for i, d in df_disgust.iterrows():
    if i not in df_disgust_reducedIndx:
        os.remove(os.path.join(d["folderName"], d["imageName"]))
        count += 1
print("Total number of images removed = "+str(count))

"""### Fear"""

anime_fear = glob.glob("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Animated/Fear/*.png")
print("Number of images in Fear emotion = "+str(len(anime_fear)))

anime_fear_folderName = [os.path.dirname(x) for x in anime_fear]
anime_fear_imageName = [os.path.basename(x) for x in anime_fear]
anime_fear_emotion = [["Fear"]*len(anime_fear)][0]
anime_fear_label = [3]*len(anime_fear)

len(anime_fear_folderName), len(anime_fear_imageName), len(anime_fear_emotion), len(anime_fear_label)

df_fear = pd.DataFrame()
df_fear["folderName"] = anime_fear_folderName
df_fear["imageName"] = anime_fear_imageName
df_fear["Emotion"] = anime_fear_emotion
df_fear["Labels"] = anime_fear_label
df_fear.head()

df_fear = df_fear.sample(frac = 1.0) #shuffling dataframe
df_fear_reduced = df_fear.sample(n = 220)  #taking only 1300 random images
df_fear_reduced.shape

#removing all the extra images from storage
df_fear_reducedIndx = df_fear_reduced.index
count = 0
for i, d in df_fear.iterrows():
    if i not in df_fear_reducedIndx:
        os.remove(os.path.join(d["folderName"], d["imageName"]))
        count += 1
print("Total number of images removed = "+str(count))

"""### Happy"""

anime_happy = glob.glob("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Animated/Happy/*.png")
print("Number of images in Happy emotion = "+str(len(anime_happy)))

anime_happy_folderName = [os.path.dirname(x) for x in anime_happy]
anime_happy_imageName = [os.path.basename(x) for x in anime_happy]
anime_happy_emotion = [["Happy"]*len(anime_happy)][0]
anime_happy_label = [4]*len(anime_happy)

len(anime_happy_folderName), len(anime_happy_imageName), len(anime_happy_emotion), len(anime_happy_label)

df_happy = pd.DataFrame()
df_happy["folderName"] = anime_happy_folderName
df_happy["imageName"] = anime_happy_imageName
df_happy["Emotion"] = anime_happy_emotion
df_happy["Labels"] = anime_happy_label
df_happy.head()

df_happy = df_happy.sample(frac = 1.0) #shuffling dataframe
df_happy_reduced = df_happy.sample(n = 300)  #taking only 1300 random images
df_happy_reduced.shape

#removing all the extra images from storage
df_happy_reducedIndx = df_happy_reduced.index
count = 0
for i, d in df_happy.iterrows():
    if i not in df_happy_reducedIndx:
        os.remove(os.path.join(d["folderName"], d["imageName"]))
        count += 1
print("Total number of images removed = "+str(count))

"""### Neutral"""

anime_neutral = glob.glob("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Animated/Neutral/*.png")
print("Number of images in Neutral emotion = "+str(len(anime_neutral)))

anime_neutral_folderName = [os.path.dirname(x) for x in anime_neutral]
anime_neutral_imageName = [os.path.basename(x) for x in anime_neutral]
anime_neutral_emotion = [["Neutral"]*len(anime_neutral)][0]
anime_neutral_label = [5]*len(anime_neutral)

len(anime_neutral_folderName), len(anime_neutral_imageName), len(anime_neutral_emotion), len(anime_neutral_label)

df_neutral = pd.DataFrame()
df_neutral["folderName"] = anime_neutral_folderName
df_neutral["imageName"] = anime_neutral_imageName
df_neutral["Emotion"] = anime_neutral_emotion
df_neutral["Labels"] = anime_neutral_label
df_neutral.head()

df_neutral = df_neutral.sample(frac = 1.0) #shuffling dataframe
df_neutral_reduced = df_neutral.sample(n = 300)  #taking only 1300 random images
df_neutral_reduced.shape

#removing all the extra images from storage
df_neutral_reducedIndx = df_neutral_reduced.index
count = 0
for i, d in df_neutral.iterrows():
    if i not in df_neutral_reducedIndx:
        os.remove(os.path.join(d["folderName"], d["imageName"]))
        count += 1
print("Total number of images removed = "+str(count))

"""### Sad"""

anime_sad = glob.glob("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Animated/Sad/*.png")
print("Number of images in Sad emotion = "+str(len(anime_sad)))

anime_sad_folderName = [os.path.dirname(x) for x in anime_sad]
anime_sad_imageName = [os.path.basename(x) for x in anime_sad]
anime_sad_emotion = [["Sad"]*len(anime_sad)][0]
anime_sad_label = [6]*len(anime_sad)

len(anime_sad_folderName), len(anime_sad_imageName), len(anime_sad_emotion), len(anime_sad_label)

df_sad = pd.DataFrame()
df_sad["folderName"] = anime_sad_folderName
df_sad["imageName"] = anime_sad_imageName
df_sad["Emotion"] = anime_sad_emotion
df_sad["Labels"] = anime_sad_label
df_sad.head()

df_sad = df_sad.sample(frac = 1.0) #shuffling dataframe
df_sad_reduced = df_sad.sample(n = 290)  #taking only 1300 random images
df_sad_reduced.shape

#removing all the extra images from storage
df_sad_reducedIndx = df_sad_reduced.index
count = 0
for i, d in df_sad.iterrows():
    if i not in df_sad_reducedIndx:
        os.remove(os.path.join(d["folderName"], d["imageName"]))
        count += 1
print("Total number of images removed = "+str(count))

"""### Surprise"""

anime_surprise = glob.glob("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Animated/Surprise/*.png")
print("Number of images in Surprise emotion = "+str(len(anime_surprise)))

anime_surprise_folderName = [os.path.dirname(x) for x in anime_surprise]
anime_surprise_imageName = [os.path.basename(x) for x in anime_surprise]
anime_surprise_emotion = [["Surprise"]*len(anime_surprise)][0]
anime_surprise_label = [7]*len(anime_surprise)

len(anime_surprise_folderName), len(anime_surprise_imageName), len(anime_surprise_emotion), len(anime_surprise_label)

df_surprise = pd.DataFrame()
df_surprise["folderName"] = anime_surprise_folderName
df_surprise["imageName"] = anime_surprise_imageName
df_surprise["Emotion"] = anime_surprise_emotion
df_surprise["Labels"] = anime_surprise_label
df_surprise.head()

df_surprise = df_surprise.sample(frac = 1.0) #shuffling dataframe
df_surprise_reduced = df_surprise.sample(n = 360)  #taking only 1300 random images
df_surprise_reduced.shape

#removing all the extra images from storage
df_surprise_reducedIndx = df_surprise_reduced.index
count = 0
for i, d in df_surprise.iterrows():
    if i not in df_surprise_reducedIndx:
        os.remove(os.path.join(d["folderName"], d["imageName"]))
        count += 1
print("Total number of images removed = "+str(count))

"""### Concatenating all Datafames"""

frames = [df_angry_reduced, df_disgust_reduced, df_fear_reduced, df_happy_reduced, df_neutral_reduced, df_sad_reduced, df_surprise_reduced]
Final_Animated = pd.concat(frames)
Final_Animated.shape

Final_Animated.reset_index(inplace = True, drop = True)
Final_Animated = Final_Animated.sample(frac = 1.0)   #shuffling the dataframe
Final_Animated.reset_index(inplace = True, drop = True)
Final_Animated.head()

"""## 6. Train, CV and Test Split for Animated Images"""

df_anime_train_data, df_anime_test = train_test_split(Final_Animated, stratify=Final_Animated["Labels"], test_size = 0.131868)
df_anime_train, df_anime_cv = train_test_split(df_anime_train_data, stratify=df_anime_train_data["Labels"], test_size = 0.088607)
df_anime_train.shape, df_anime_cv.shape, df_anime_test.shape

df_anime_train.reset_index(inplace = True, drop = True)
df_anime_train.to_pickle("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Dataframes/Animated/df_anime_train.pkl")

df_anime_cv.reset_index(inplace = True, drop = True)
df_anime_cv.to_pickle("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Dataframes/Animated/df_anime_cv.pkl")

df_anime_test.reset_index(inplace = True, drop = True)
df_anime_test.to_pickle("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Dataframes/Animated/df_anime_test.pkl")

df_anime_train = pd.read_pickle("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Dataframes/Animated/df_anime_train.pkl")
df_anime_train.head()
#df_anime_train.shape

df_anime_train.shape

df_anime_cv = pd.read_pickle("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Dataframes/Animated/df_anime_cv.pkl")
df_anime_cv.head()

df_anime_cv.shape

df_anime_test = pd.read_pickle("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Dataframes/Animated/df_anime_test.pkl")
df_anime_test.head()

df_anime_test.shape

"""## 7. Analysing Data of Animated Images
### Distribution of class labels in Train, CV and Test
"""

df_temp_train = df_anime_train.sort_values(by = "Labels", inplace = False)
df_temp_cv = df_anime_cv.sort_values(by = "Labels", inplace = False)
df_temp_test = df_anime_test.sort_values(by = "Labels", inplace = False)

TrainData_distribution = df_anime_train["Emotion"].value_counts().sort_index()
CVData_distribution = df_anime_cv["Emotion"].value_counts().sort_index()
TestData_distribution = df_anime_test["Emotion"].value_counts().sort_index()

TrainData_distribution_sorted = sorted(TrainData_distribution.items(), key = lambda d: d[1], reverse = True)
CVData_distribution_sorted = sorted(CVData_distribution.items(), key = lambda d: d[1], reverse = True)
TestData_distribution_sorted = sorted(TestData_distribution.items(), key = lambda d: d[1], reverse = True)

fig = plt.figure(figsize = (10, 6))
ax = fig.add_axes([0,0,1,1])
ax.set_title("Count of each Emotion in Train Data", fontsize = 20)
sns.countplot(x = "Emotion", data = df_temp_train)
plt.grid()
for i in ax.patches:
    ax.text(x = i.get_x() + 0.185, y = i.get_height()+1.6, s = str(i.get_height()), fontsize = 20, color = "grey")
plt.xlabel("")
plt.ylabel("Count", fontsize = 15)
plt.tick_params(labelsize = 15)
plt.xticks(rotation = 40)
plt.show()

for i in TrainData_distribution_sorted:
    print("Number of training data points in class "+str(i[0])+" = "+str(i[1])+ "("+str(np.round(((i[1]/df_temp_train.shape[0])*100), 4))+"%)")

print("-"*80)

fig = plt.figure(figsize = (10, 6))
ax = fig.add_axes([0,0,1,1])
ax.set_title("Count of each Emotion in Validation Data", fontsize = 20)
sns.countplot(x = "Emotion", data = df_temp_cv)
plt.grid()
for i in ax.patches:
    ax.text(x = i.get_x() + 0.21, y = i.get_height()+0.3, s = str(i.get_height()), fontsize = 20, color = "grey")
plt.xlabel("")
plt.ylabel("Count", fontsize = 15)
plt.tick_params(labelsize = 15)
plt.xticks(rotation = 40)
plt.show()

for i in CVData_distribution_sorted:
    print("Number of training data points in class "+str(i[0])+" = "+str(i[1])+ "("+str(np.round(((i[1]/df_temp_cv.shape[0])*100), 4))+"%)")

print("-"*80)

fig = plt.figure(figsize = (10, 6))
ax = fig.add_axes([0,0,1,1])
ax.set_title("Count of each Emotion in Test Data", fontsize = 20)
sns.countplot(x = "Emotion", data = df_temp_test)
plt.grid()
for i in ax.patches:
    ax.text(x = i.get_x() + 0.21, y = i.get_height()+0.3, s = str(i.get_height()), fontsize = 20, color = "grey")
plt.xlabel("")
plt.ylabel("Count", fontsize = 15)
plt.tick_params(labelsize = 15)
plt.xticks(rotation = 40)
plt.show()

for i in TestData_distribution_sorted:
    print("Number of training data points in class "+str(i[0])+" = "+str(i[1])+ "("+str(np.round(((i[1]/df_temp_test.shape[0])*100), 4))+"%)")

"""## 8. Pre-Processing Animated Images

### 8.1 Converting all the images to grayscale and save them
"""

def convt_to_gray(df):
    count = 0
    for i in range(len(df)):
        path1 = df["folderName"][i]
        path2 = df["imageName"][i]
        img = cv2.imread(os.path.join(path1, path2))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(path1, path2), gray)
        count += 1
    print("Total number of images converted and saved = "+str(count))

convt_to_gray(df_anime_train)

convt_to_gray(df_anime_cv)

convt_to_gray(df_anime_test)

"""### 8.2 Crop the image then resize them then save them."""

def change_image(df):
    count = 0
    for i, d in df.iterrows():
        img = cv2.imread(os.path.join(d["folderName"], d["imageName"]))
        face_clip = img[40:240, 35:225]         #cropping the face in image
        face_resized = cv2.resize(face_clip, (350, 350))
        cv2.imwrite(os.path.join(d["folderName"], d["imageName"]), face_resized) #resizing and saving the image
        count += 1
    print("Total number of images cropped and resized = {}".format(count))

change_image(df_anime_train)

change_image(df_anime_cv)

change_image(df_anime_test)

"""## 9. Combining train data of both Animated and Human images

Remember, that here we have combined only the train images of both human and animated so that we can train our model on both human and animated images. However, we have kept CV and test images of both human and animated separate so that we can cross validation our results on both human and animated images separately. At the same time we will also be able to test the efficiency of our model separately on human and animated images. By this we will get to know that how well our model is performing on human and animated images separately.
"""

frames = [df_human_train, df_anime_train]
combined_train = pd.concat(frames)
combined_train.shape

combined_train = combined_train.sample(frac = 1.0)  #shuffling the dataframe
combined_train.reset_index(inplace = True, drop = True)
combined_train.to_pickle("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Dataframes/combined_train.pkl")

"""## 10. Creating bottleneck features from VGG-16 model. Here, we are using Transfer learning."""

Train_Combined = pd.read_pickle("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Dataframes/combined_train.pkl")
CV_Humans = pd.read_pickle("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Dataframes/Human/df_human_cv.pkl")
CV_Animated = pd.read_pickle("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Dataframes/Animated/df_anime_cv.pkl")
Test_Humans = pd.read_pickle("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Dataframes/Human/df_human_test.pkl")
Test_Animated = pd.read_pickle("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Dataframes/Animated/df_anime_test.pkl")

Train_Combined.shape, CV_Humans.shape, CV_Animated.shape, Test_Humans.shape, Test_Animated.shape

TrainCombined_batch_pointer = 0
CVHumans_batch_pointer = 0
CVAnimated_batch_pointer = 0
TestHumans_batch_pointer = 0
TestAnimated_batch_pointer = 0

"""### 10.1 Bottleneck features for CombinedTrain Data"""

#TrainCombined_Labels = pd.get_dummies(Train_Combined["Labels"]).as_matrix()
TrainCombined_Labels = pd.get_dummies(Train_Combined["Labels"]).values
TrainCombined_Labels.shape

def loadCombinedTrainBatch(batch_size):
    global TrainCombined_batch_pointer
    batch_images = []
    batch_labels = []
    for i in range(batch_size):
        path1 = Train_Combined.iloc[TrainCombined_batch_pointer + i]["folderName"]
        path2 = Train_Combined.iloc[TrainCombined_batch_pointer + i]["imageName"]
        read_image = cv2.imread(os.path.join(path1, path2))
        read_image_final = read_image/255.0  #here, we are normalizing the images
        batch_images.append(read_image_final)
        
        batch_labels.append(TrainCombined_Labels[TrainCombined_batch_pointer + i]) #appending corresponding labels
        
    TrainCombined_batch_pointer += batch_size
        
    return np.array(batch_images), np.array(batch_labels)

#creating bottleneck features for train data using VGG-16- Image-net model
model = VGG16(weights='imagenet', include_top=False)
SAVEDIR = "/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Bottleneck_Features/Bottleneck_CombinedTrain/"
SAVEDIR_LABELS = "/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Bottleneck_Features/CombinedTrain_Labels/"
batch_size = 10
for i in range(int(len(Train_Combined)/batch_size)):
    x, y = loadCombinedTrainBatch(batch_size)
    print("Batch {} loaded".format(i+1))
    
    np.save(os.path.join(SAVEDIR_LABELS, "bottleneck_labels_{}".format(i+1)), y)
    
    print("Creating bottleneck features for batch {}". format(i+1))
    bottleneck_features = model.predict(x)
    np.save(os.path.join(SAVEDIR, "bottleneck_{}".format(i+1)), bottleneck_features)
    print("Bottleneck features for batch {} created and saved\n".format(i+1))

"""### 10.2 Bottleneck features for CV Human"""

CVHumans_Labels = pd.get_dummies(CV_Humans["Labels"]).values
CVHumans_Labels.shape

def loadCVHumanBatch(batch_size):
    global CVHumans_batch_pointer
    batch_images = []
    batch_labels = []
    for i in range(batch_size):
        path1 = CV_Humans.iloc[CVHumans_batch_pointer + i]["folderName"]
        path2 = CV_Humans.iloc[CVHumans_batch_pointer + i]["imageName"]
        read_image = cv2.imread(os.path.join(path1, path2))
        read_image_final = read_image/255.0  #here, we are normalizing the images
        batch_images.append(read_image_final)
        
        batch_labels.append(CVHumans_Labels[CVHumans_batch_pointer + i]) #appending corresponding labels
        
    CVHumans_batch_pointer += batch_size
        
    return np.array(batch_images), np.array(batch_labels)

#creating bottleneck features for CV Human data using VGG-16- Image-net model
model = VGG16(weights='imagenet', include_top=False)
SAVEDIR = "/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Bottleneck_Features/Bottleneck_CVHumans/"
SAVEDIR_LABELS = "/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Bottleneck_Features/CVHumans_Labels/"
batch_size = 10
for i in range(int(len(CV_Humans)/batch_size)):
    x, y = loadCVHumanBatch(batch_size)
    print("Batch {} loaded".format(i+1))
    
    np.save(os.path.join(SAVEDIR_LABELS, "bottleneck_labels_{}".format(i+1)), y)
    
    print("Creating bottleneck features for batch {}". format(i+1))
    bottleneck_features = model.predict(x)
    np.save(os.path.join(SAVEDIR, "bottleneck_{}".format(i+1)), bottleneck_features)
    print("Bottleneck features for batch {} created and saved\n".format(i+1))

"""### 10.3 Bottleneck features for CV Animated"""

#CVAnimated_Labels = pd.get_dummies(CV_Animated["Labels"]).as_matrix()
CVAnimated_Labels = pd.get_dummies(CV_Animated["Labels"]).values
CVAnimated_Labels.shape

def loadCVAnimatedBatch(batch_size):
    global CVAnimated_batch_pointer
    batch_images = []
    batch_labels = []
    for i in range(batch_size):
        path1 = CV_Animated.iloc[CVAnimated_batch_pointer + i]["folderName"]
        path2 = CV_Animated.iloc[CVAnimated_batch_pointer + i]["imageName"]
        read_image = cv2.imread(os.path.join(path1, path2))
        read_image_final = read_image/255.0  #here, we are normalizing the images
        batch_images.append(read_image_final)
        
        batch_labels.append(CVAnimated_Labels[CVAnimated_batch_pointer + i]) #appending corresponding labels
        
    CVAnimated_batch_pointer += batch_size
        
    return np.array(batch_images), np.array(batch_labels)

#creating bottleneck features for CV Animated data using VGG-16- Image-net model
model = VGG16(weights='imagenet', include_top=False)
SAVEDIR = "/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Bottleneck_Features/Bottleneck_CVAnimated/"
SAVEDIR_LABELS = "/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Bottleneck_Features/CVAnimated_Labels/"
batch_size = 10
for i in range(int(len(CV_Animated)/batch_size)):
    x, y = loadCVAnimatedBatch(batch_size)
    print("Batch {} loaded".format(i+1))
    
    np.save(os.path.join(SAVEDIR_LABELS, "bottleneck_labels_{}".format(i+1)), y)
    
    print("Creating bottleneck features for batch {}". format(i+1))
    bottleneck_features = model.predict(x)
    np.save(os.path.join(SAVEDIR, "bottleneck_{}".format(i+1)), bottleneck_features)
    print("Bottleneck features for batch {} created and saved\n".format(i+1))

"""### 10.4 Bottleneck Features for Test Human Data"""

#TestHuman_Labels = pd.get_dummies(Test_Humans["Labels"]).as_matrix()
TestHuman_Labels = pd.get_dummies(Test_Humans["Labels"]).values
TestHuman_Labels.shape

def loadTestHumansBatch(batch_size):
    global TestHumans_batch_pointer
    batch_images = []
    batch_labels = []
    for i in range(batch_size):
        path1 = Test_Humans.iloc[TestHumans_batch_pointer + i]["folderName"]
        path2 = Test_Humans.iloc[TestHumans_batch_pointer + i]["imageName"]
        read_image = cv2.imread(os.path.join(path1, path2))
        read_image_final = read_image/255.0  #here, we are normalizing the images
        batch_images.append(read_image_final)
        
        batch_labels.append(TestHuman_Labels[TestHumans_batch_pointer + i]) #appending corresponding labels
        
    TestHumans_batch_pointer += batch_size
        
    return np.array(batch_images), np.array(batch_labels)

#creating bottleneck features for Test Humans data using VGG-16- Image-net model
model = VGG16(weights='imagenet', include_top=False)
SAVEDIR = "/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Bottleneck_Features/Bottleneck_TestHumans/"
SAVEDIR_LABELS = "/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Bottleneck_Features/TestHumans_Labels/"
batch_size = 10
for i in range(int(len(Test_Humans)/batch_size)):
    x, y = loadTestHumansBatch(batch_size)
    print("Batch {} loaded".format(i+1))
    
    np.save(os.path.join(SAVEDIR_LABELS, "bottleneck_labels_{}".format(i+1)), y)
    
    print("Creating bottleneck features for batch {}". format(i+1))
    bottleneck_features = model.predict(x)
    np.save(os.path.join(SAVEDIR, "bottleneck_{}".format(i+1)), bottleneck_features)
    print("Bottleneck features for batch {} created and saved\n".format(i+1))

#leftover_points = len(Test_Humans) - TestHumans_batch_pointer
#x, y = loadTestHumansBatch(leftover_points)
#np.save(os.path.join(SAVEDIR_LABELS, "bottleneck_labels_{}".format(int(len(Test_Humans)/batch_size) + 1)), y)
#bottleneck_features = model.predict(x)
#np.save(os.path.join(SAVEDIR, "bottleneck_{}".format(int(len(Test_Humans)/batch_size) + 1)), bottleneck_features)

"""### 10.5 Bottleneck Features for Test Animated Data"""

#TestAnimated_Labels = pd.get_dummies(Test_Animated["Labels"]).as_matrix()
TestAnimated_Labels = pd.get_dummies(Test_Animated["Labels"]).values
TestAnimated_Labels.shape

def loadTestAnimatedBatch(batch_size):
    global TestAnimated_batch_pointer
    batch_images = []
    batch_labels = []
    for i in range(batch_size):
        path1 = Test_Animated.iloc[TestAnimated_batch_pointer + i]["folderName"]
        path2 = Test_Animated.iloc[TestAnimated_batch_pointer + i]["imageName"]
        read_image = cv2.imread(os.path.join(path1, path2))
        read_image_final = read_image/255.0  #here, we are normalizing the images
        batch_images.append(read_image_final)
        
        batch_labels.append(TestAnimated_Labels[TestAnimated_batch_pointer + i]) #appending corresponding labels
        
    TestAnimated_batch_pointer += batch_size
        
    return np.array(batch_images), np.array(batch_labels)

#creating bottleneck features for Test Animated data using VGG-16- Image-net model
model = VGG16(weights='imagenet', include_top=False)
SAVEDIR = "/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Bottleneck_Features/Bottleneck_TestAnimated/"
SAVEDIR_LABELS = "/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Bottleneck_Features/TestAnimated_Labels/"
batch_size = 10
for i in range(int(len(Test_Animated)/batch_size)):
    x, y = loadTestAnimatedBatch(batch_size)
    print("Batch {} loaded".format(i+1))
    
    np.save(os.path.join(SAVEDIR_LABELS, "bottleneck_labels_{}".format(i+1)), y)
    
    print("Creating bottleneck features for batch {}". format(i+1))
    bottleneck_features = model.predict(x)
    np.save(os.path.join(SAVEDIR, "bottleneck_{}".format(i+1)), bottleneck_features)
    print("Bottleneck features for batch {} created and saved\n".format(i+1))

"""## 11. Modelling & Training"""

no_of_classes = 7

#model architecture
def model(input_shape):
    model = Sequential()
        
    model.add(Dense(512, activation='relu', input_dim = input_shape))
    model.add(Dropout(0.1))
    
    model.add(Dense(256, activation='relu'))
    
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_dim = no_of_classes, activation='softmax')) 
    
    return model

#training the model
SAVEDIR_COMB_TRAIN = "/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Bottleneck_Features/Bottleneck_CombinedTrain/"
SAVEDIR_COMB_TRAIN_LABELS = "/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Bottleneck_Features/CombinedTrain_Labels/"

SAVEDIR_CV_HUMANS = "/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Bottleneck_Features/Bottleneck_CVHumans/"
SAVEDIR_CV_HUMANS_LABELS = "/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Bottleneck_Features/CVHumans_Labels/"

SAVEDIR_CV_ANIME = "/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Bottleneck_Features/Bottleneck_CVAnimated/"
SAVEDIR_CV_ANIME_LABELS =  "/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Bottleneck_Features/CVAnimated_Labels/"

SAVER = "/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Model_Save/"

input_shape = 10*10*512   #this is the shape of bottleneck feature of each image which comes after passing the image through VGG-16

model = model(input_shape)
# model.load_weights(os.path.join(SAVER, "model.h5"))
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = "adam", metrics = ["accuracy"])

epochs = 20
batch_size = 10
step = 0
combTrain_bottleneck_files = int(len(Train_Combined) / batch_size)
CVHuman_bottleneck_files = int(len(CV_Humans) / batch_size)
CVAnime_bottleneck_files = int(len(CV_Animated) / batch_size)
epoch_number, CombTrain_loss, CombTrain_acc, CVHuman_loss, CVHuman_acc, CVAnime_loss, CVAnime_acc = [], [], [], [], [], [], []

for epoch in range(epochs):
    avg_epoch_CombTr_loss, avg_epoch_CombTr_acc, avg_epoch_CVHum_loss, avg_epoch_CVHum_acc, avg_epoch_CVAnime_loss, avg_epoch_CVAnime_acc = 0, 0, 0, 0, 0, 0
    epoch_number.append(epoch + 1)
    
    for i in range(combTrain_bottleneck_files):
        
        step += 1
        
        #loading batch of train bottleneck features for training MLP.
        X_CombTrain_load = np.load(os.path.join(SAVEDIR_COMB_TRAIN, "bottleneck_{}.npy".format(i+1)))
        X_CombTrain = X_CombTrain_load.reshape(X_CombTrain_load.shape[0], X_CombTrain_load.shape[1]*X_CombTrain_load.shape[2]*X_CombTrain_load.shape[3])
        Y_CombTrain = np.load(os.path.join(SAVEDIR_COMB_TRAIN_LABELS, "bottleneck_labels_{}.npy".format(i+1)))
        
        #loading batch of Human CV bottleneck features for cross-validation.
        X_CVHuman_load = np.load(os.path.join(SAVEDIR_CV_HUMANS, "bottleneck_{}.npy".format((i % CVHuman_bottleneck_files) + 1)))
        X_CVHuman = X_CVHuman_load.reshape(X_CVHuman_load.shape[0], X_CVHuman_load.shape[1]*X_CVHuman_load.shape[2]*X_CVHuman_load.shape[3])
        Y_CVHuman = np.load(os.path.join(SAVEDIR_CV_HUMANS_LABELS, "bottleneck_labels_{}.npy".format((i % CVHuman_bottleneck_files) + 1)))
        
        #loading batch of animated CV bottleneck features for cross-validation.
        X_CVAnime_load = np.load(os.path.join(SAVEDIR_CV_ANIME, "bottleneck_{}.npy".format((i % CVAnime_bottleneck_files) + 1)))
        X_CVAnime = X_CVAnime_load.reshape(X_CVAnime_load.shape[0], X_CVAnime_load.shape[1]*X_CVAnime_load.shape[2]*X_CVAnime_load.shape[3])
        Y_CVAnime = np.load(os.path.join(SAVEDIR_CV_ANIME_LABELS, "bottleneck_labels_{}.npy".format((i % CVAnime_bottleneck_files) + 1)))
        
        CombTrain_Loss, CombTrain_Accuracy = model.train_on_batch(X_CombTrain, Y_CombTrain) #train the model on batch
        CVHuman_Loss, CVHuman_Accuracy = model.test_on_batch(X_CVHuman, Y_CVHuman) #cross validate the model on CV Human batch
        CVAnime_Loss, CVAnime_Accuracy = model.test_on_batch(X_CVAnime, Y_CVAnime) #cross validate the model on CV Animated batch
        
        print("Epoch: {}, Step: {}, CombTr_Loss: {}, CombTr_Acc: {}, CVHum_Loss: {}, CVHum_Acc: {}, CVAni_Loss: {}, CVAni_Acc: {}".format(epoch+1, step, np.round(float(CombTrain_Loss), 2), np.round(float(CombTrain_Accuracy), 2), np.round(float(CVHuman_Loss), 2), np.round(float(CVHuman_Accuracy), 2), np.round(float(CVAnime_Loss), 2), np.round(float(CVAnime_Accuracy), 2)))
        
        avg_epoch_CombTr_loss += CombTrain_Loss / combTrain_bottleneck_files
        avg_epoch_CombTr_acc += CombTrain_Accuracy / combTrain_bottleneck_files
        avg_epoch_CVHum_loss += CVHuman_Loss / combTrain_bottleneck_files
        avg_epoch_CVHum_acc += CVHuman_Accuracy / combTrain_bottleneck_files
        avg_epoch_CVAnime_loss += CVAnime_Loss / combTrain_bottleneck_files
        avg_epoch_CVAnime_acc += CVAnime_Accuracy / combTrain_bottleneck_files
        
    print("Avg_CombTrain_Loss: {}, Avg_CombTrain_Acc: {}, Avg_CVHum_Loss: {}, Avg_CVHum_Acc: {}, Avg_CVAnime_Loss: {}, Avg_CVAnime_Acc: {}".format(np.round(float(avg_epoch_CombTr_loss), 2), np.round(float(avg_epoch_CombTr_acc), 2), np.round(float(avg_epoch_CVHum_loss), 2), np.round(float(avg_epoch_CVHum_acc), 2), np.round(float(avg_epoch_CVAnime_loss), 2), np.round(float(avg_epoch_CVAnime_acc), 2)))

    CombTrain_loss.append(avg_epoch_CombTr_loss)
    CombTrain_acc.append(avg_epoch_CombTr_acc)
    CVHuman_loss.append(avg_epoch_CVHum_loss)
    CVHuman_acc.append(avg_epoch_CVHum_acc)
    CVAnime_loss.append(avg_epoch_CVAnime_loss)
    CVAnime_acc.append(avg_epoch_CVAnime_acc)
    
    model.save(os.path.join(SAVER, "model.h5"))  #saving the model on each epoc
    model.save_weights(os.path.join(SAVER, "model_weights.h5")) #saving the weights of model on each epoch
    print("Model and weights saved at epoch {}".format(epoch + 1))
          
log_frame = pd.DataFrame(columns = ["Epoch", "Comb_Train_Loss", "Comb_Train_Accuracy", "CVHuman_Loss", "CVHuman_Accuracy", "CVAnime_Loss", "CVAnime_Accuracy"])
log_frame["Epoch"] = epoch_number
log_frame["Comb_Train_Loss"] = CombTrain_loss
log_frame["Comb_Train_Accuracy"] = CombTrain_acc
log_frame["CVHuman_Loss"] = CVHuman_loss
log_frame["CVHuman_Accuracy"] = CVHuman_acc
log_frame["CVAnime_Loss"] = CVAnime_loss
log_frame["CVAnime_Accuracy"] = CVAnime_acc
log_frame.to_csv("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Logs/Log.csv", index = False)

log = pd.read_csv("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Logs/Log.csv")
log

def plotting(epoch, train_loss, CVHuman_loss, CVAnimated_loss, title):
    fig, axes = plt.subplots(1,1, figsize = (12, 8))
    axes.plot(epoch, train_loss, color = 'red', label = "Train")
    axes.plot(epoch, CVHuman_loss, color = 'blue', label = "CV_Human")
    axes.plot(epoch, CVAnimated_loss, color = 'green', label = "CV_Animated")
    axes.set_title(title, fontsize = 25)
    axes.set_xlabel("Epochs", fontsize = 20)
    axes.set_ylabel("Loss", fontsize = 20)
    axes.grid()
    axes.legend(fontsize = 20)

plotting(list(log["Epoch"]), list(log["Comb_Train_Loss"]), list(log["CVHuman_Loss"]), list(log["CVAnime_Loss"]), "EPOCH VS LOSS")

def plotting(epoch, train_acc, CVHuman_acc, CVAnimated_acc, title):
    fig, axes = plt.subplots(1,1, figsize = (12, 8))
    axes.plot(epoch, train_acc, color = 'red', label = "Train_Accuracy")
    axes.plot(epoch, CVHuman_acc, color = 'blue', label = "CV_Human_Accuracy")
    axes.plot(epoch, CVAnimated_acc, color = 'green', label = "CV_Animated_Accuracy")
    axes.set_title(title, fontsize = 25)
    axes.set_xlabel("Epochs", fontsize = 20)
    axes.set_ylabel("Accuracy", fontsize = 20)
    axes.grid()
    axes.legend(fontsize = 20)

plotting(list(log["Epoch"]), list(log["Comb_Train_Accuracy"]), list(log["CVHuman_Accuracy"]), list(log["CVAnime_Accuracy"]), "EPOCH VS ACCURACY")

"""## 12. Checking Test Accuracy"""

def print_confusionMatrix(Y_TestLabels, PredictedLabels):
    confusionMatx = confusion_matrix(Y_TestLabels, PredictedLabels)
    
    precision = confusionMatx/confusionMatx.sum(axis = 0)
    
    recall = (confusionMatx.T/confusionMatx.sum(axis = 1)).T
    
    sns.set(font_scale=1.5)
    
    # confusionMatx = [[1, 2],
    #                  [3, 4]]
    # confusionMatx.T = [[1, 3],
    #                   [2, 4]]
    # confusionMatx.sum(axis = 1)  axis=0 corresponds to columns and axis=1 corresponds to rows in two diamensional array
    # confusionMatx.sum(axix =1) = [[3, 7]]
    # (confusionMatx.T)/(confusionMatx.sum(axis=1)) = [[1/3, 3/7]
    #                                                  [2/3, 4/7]]

    # (confusionMatx.T)/(confusionMatx.sum(axis=1)).T = [[1/3, 2/3]
    #                                                    [3/7, 4/7]]
    # sum of row elements = 1
    
    labels = ["ANGRY", "DISGUST", "FEAR", "HAPPY", "NEUTRAL", "SAD", "SURPRISE"]
    
    plt.figure(figsize=(16,7))
    sns.heatmap(confusionMatx, cmap = "Blues", annot = True, fmt = ".1f", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix", fontsize = 30)
    plt.xlabel('Predicted Class', fontsize = 20)
    plt.ylabel('Original Class', fontsize = 20)
    plt.tick_params(labelsize = 15)
    plt.xticks(rotation = 90)
    plt.show()
    
    print("-"*125)
    
    plt.figure(figsize=(16,7))
    sns.heatmap(precision, cmap = "Blues", annot = True, fmt = ".2f", xticklabels=labels, yticklabels=labels)
    plt.title("Precision Matrix", fontsize = 30)
    plt.xlabel('Predicted Class', fontsize = 20)
    plt.ylabel('Original Class', fontsize = 20)
    plt.tick_params(labelsize = 15)
    plt.xticks(rotation = 90)
    plt.show()
    
    print("-"*125)
    
    plt.figure(figsize=(16,7))
    sns.heatmap(recall, cmap = "Blues", annot = True, fmt = ".2f", xticklabels=labels, yticklabels=labels)
    plt.title("Recall Matrix", fontsize = 30)
    plt.xlabel('Predicted Class', fontsize = 20)
    plt.ylabel('Original Class', fontsize = 20)
    plt.tick_params(labelsize = 15)
    plt.xticks(rotation = 90)
    plt.show()

"""### Test Data of Human Images"""

model = load_model("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Model_Save/model.h5")
predicted_labels = []
true_labels = []
batch_size = 10
total_files = int(len(Test_Humans) / batch_size) + 2 #here, I have added 2 because there are 30 files in Test_Humans
total_files = total_files-1
for i in range(1, total_files, 1):
    img_load = np.load("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Bottleneck_Features/Bottleneck_TestHumans/bottleneck_{}.npy".format(i))
    img_label = np.load("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Bottleneck_Features/TestHumans_Labels/bottleneck_labels_{}.npy".format(i))
    img_bundle = img_load.reshape(img_load.shape[0], img_load.shape[1]*img_load.shape[2]*img_load.shape[3])
    for j in range(img_bundle.shape[0]):
        img = img_bundle[j]
        img = img.reshape(1, img_bundle.shape[1])
        pred = model.predict(img)
        predicted_labels.append(pred[0].argmax())
        true_labels.append(img_label[j].argmax())
acc = accuracy_score(true_labels, predicted_labels)
print("Accuracy on Human Test Data = {}%".format(np.round(float(acc*100), 2)))

print_confusionMatrix(true_labels, predicted_labels)

"""### Test Data of Animated Images"""

model = load_model("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Model_Save/model.h5")
predicted_labels = []
true_labels = []
batch_size = 10
total_files = int(len(Test_Animated) / batch_size) + 1
for i in range(1, total_files, 1):
    img_load = np.load("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Bottleneck_Features/Bottleneck_TestAnimated/bottleneck_{}.npy".format(i))
    img_label = np.load("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Bottleneck_Features/TestAnimated_Labels/bottleneck_labels_{}.npy".format(i))
    img_bundle = img_load.reshape(img_load.shape[0], img_load.shape[1]*img_load.shape[2]*img_load.shape[3])
    for j in range(img_bundle.shape[0]):
        img = img_bundle[j]
        img = img.reshape(1, img_bundle.shape[1])
        pred = model.predict(img)
        predicted_labels.append(pred[0].argmax())
        true_labels.append(img_label[j].argmax())
acc = accuracy_score(true_labels, predicted_labels)
print("Accuracy on Animated Test Data = {}%".format(np.round(float(acc*100), 2)))

print_confusionMatrix(true_labels, predicted_labels)

"""## 13. Testing on Real World with Still Images"""

# Now for testing the model on real world images we have to follow all of the same steps which we have done on our training, CV
# and test images. Like here we have to first pre-preocess our images then create its VGG-16 bottleneck features then pass those 
# bottleneck features through our own MLP model for prediction.
# Steps are as follows:
# 1. Read the image, convert it to grayscale and save it.
# 2. Read that grayscale saved image, the detect face in it using HAAR cascade.
# 3. Crop the image to the detected face and resize it to 350*350 and save the image.
# 4. Read that processed cropped-resized image, then reshape it and normalize it.
# 5. Then feed that image to VGG-16 and create bottleneck features of that image and then reshape it.
# 6. Then use our own model for final prediction of expression.

EMOTION_DICT = {1:"ANGRY", 2:"DISGUST", 3:"FEAR", 4:"HAPPY", 5:"NEUTRAL", 6:"SAD", 7:"SURPRISE"}
model_VGG = VGG16(weights='imagenet', include_top=False)
model_top = load_model("/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Model_Save/model.h5")

def make_prediction(path):
    #converting image to gray scale and save it
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    path2 = '/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/Data/Test_Images/temp.bmp'
    cv2.imwrite(path2, gray)
    
    
    #detect face in image, crop it then resize it then save it
    face_cascade = cv2.CascadeClassifier('/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/haarcascade_frontalface_default.xml') 
    #/content/drive/My Drive/CNN/FacialEmotion/RealTimeFacialRecognition/haarcascade_frontalface_default.xml
    img = cv2.imread(path2)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print(faces)
    if faces == ():
      cv2.imwrite(path2, cv2.resize(gray, (350, 350)))
    else:
      for (x,y,w,h) in faces: # to be modified
        face_clip = img[y:y+h, x:x+w]# to be modified
        print(x,y,w,h)
        cv2.imwrite(path2, cv2.resize(face_clip, (350, 350)))# to be modified

    #for (x,y,w,h) in faces: # to be modified

    
    
    #read the processed image then make prediction and display the result
    read_image = cv2.imread(path2)
    read_image = read_image.reshape(1, read_image.shape[0], read_image.shape[1], read_image.shape[2])
    read_image_final = read_image/255.0  #normalizing the image
    VGG_Pred = model_VGG.predict(read_image_final)  #creating bottleneck features of image using VGG-16.
    VGG_Pred = VGG_Pred.reshape(1, VGG_Pred.shape[1]*VGG_Pred.shape[2]*VGG_Pred.shape[3])
    top_pred = model_top.predict(VGG_Pred)  #making prediction from our own model.
    emotion_label = top_pred[0].argmax() + 1
    print("Predicted Expression Probabilities")
    print("ANGRY: {}\nDISGUST: {}\nFEAR: {}\nHAPPY: {}\nNEUTRAL: {}\nSAD: {}\nSURPRISE: {}\n\n".format(top_pred[0][0], top_pred[0][1], top_pred[0][2], top_pred[0][3], top_pred[0][4], top_pred[0][5], top_pred[0][6]))
    print("Dominant Probability = "+str(EMOTION_DICT[emotion_label])+": "+str(max(top_pred[0])))

"""### ANGRY

### Correct Result
"""


