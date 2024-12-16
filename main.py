import os.path
import cv2
import numpy as np
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.numpy_ops.np_math_ops import argmin
from torch.utils.tensorboard.summary import image

#load csv
data_dir = 'C:/Users\sudha\PycharmProjects\FacialExpressionRecognition/facial-emotion-recognition\images'

#defining emotion mapping
EMOTIONS = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']
#data['label'] = data['emotion'].map(EMOTIONS) #convert emotions to numeric value
IMG_SIZE = 48


def load_data():
    images = []
    labels = []

    for emotion_label, emotion in enumerate(EMOTIONS):
        emotion_dir = os.path.join(data_dir, str(emotion_label))
        if os.path.isdir(emotion_dir):
            #print(f"Directory not found for emotion {emotion}: {emotion_dir}")
            for img_file in os.listdir(emotion_dir):
                img_path = os.path.join(emotion_dir, img_file)
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) #cv2
                if img_array is not None:
                   img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                   images.append(img_resized)
                   labels.append(emotion_label)
        else:
            print(f"Directory not found for enumeration {emotion}: {emotion_dir}")
    return np.array(images), np.array(labels)


images, labels = load_data()

#splitting data into training and test
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

#normalize pixel
X_train = X_train / 255.0
X_test = X_test / 255.0

#reshaping data for CNN
X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  #(batch_size[-1], height[IMG_SIZE], width[IMG_SIZE], channels[1 for grayscale])
X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

#visualisation(this will show if images are loaded sahi se or not)
#print information
print("Number of images", len(images))
print("shape of image array", images.shape)
print("Number of labels", len(labels))

#plot of images and corresponding labels
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 20))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(EMOTIONS[labels[i]])
    plt.axis('off')
plt.show()

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

from tensorflow.keras import models, layers

#define CNN model
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),  #dropout prevents overfitting
    layers.Dense(len(EMOTIONS), activation='softmax')
])

#compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

#train the model
#from tensorflow.keras.callbacks import EarlyStopping

#early_stopping = EarlyStopping(
#    monitor='val_loss',
#    patience=5,
#    restore_best_weights=True
#)
#early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weight=True)
history = model.fit(datagen.flow(X_train, y_train, batch_size=48),
                    epochs=50, validation_data=datagen.flow(X_test, y_test))#, callbacks=[early_stopping])

#visualize model

tr_acc=history.history['accuracy']
tr_loss=history.history['loss']
val_acc=history.history['val_accuracy']
val_loss=history.history['val_loss']
index_loss=np.argmin(val_loss)
val_lowest=val_loss[index_loss]
index_acc=np.argmax(val_acc)
acc_highest=val_acc[index_acc]

Epochs=[i+1 for i in range(len(tr_acc))]
loss_label=f'best epoch = {str(index_loss+1)}'
acc_label=f'best epoch = {str(index_acc+1)}'

#plot training history

plt.figure(figsize=(20,8))
plt.style.use('fivethirtyeight')

plt.subplot(1,2,1)
plt.plot(Epochs, tr_loss, 'r', label='Training loss')
plt.plot(Epochs, val_loss, 'g', label='Validation loss')
plt.scatter(index_loss+1, val_lowest, s=150, c='blue', label=loss_label)
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(Epochs, tr_acc, 'r', label='Training accuracy')
plt.plot(Epochs, val_acc, 'g', label='Validation accuracy')
plt.scatter(index_acc+1, acc_highest, s=150, c='blue', label=acc_label)
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

#evaluate the model

#evaluate on test_set
test_loss, test_accuracy=model.evaluate(X_test,y_test,verbose=2)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

from sklearn.metrics import classification_report, confusion_matrix

#make predictions
Predictions=model.predict(X_test)
predicted_labels=np.argmax(Predictions,axis=1)
#print class reports
#print(classification_report(y_test, predicted_labels, target_names=EMOTIONS))

#generate confusion matrix
cm=confusion_matrix(y_test,predicted_labels)

import seaborn as sns
#plot confusion matrix
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTIONS, yticklabels=EMOTIONS)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

#function to display image with predicted and actual labels
def display_image(images, actual_labels, predicted_labels, emotions, num_images=10):
    plt.figure(figsize=(20,10))
    for i in range(num_images):
        plt.subplot(2,5,i+1)
        #since we have images in grayscale we need to reshape them to (48,48)
        img=images[i].reshape(IMG_SIZE,IMG_SIZE)
        plt.imshow(img, cmap='gray')
        plt.title(f"Actual:{emotions[actual_labels[i]]}\n"
                  f" Predicted:{emotions[predicted_labels[i]]}")
        plt.axis('off')
    plt.show()

#display image with actual and predicted labels
display_image(X_test, y_test, predicted_labels, EMOTIONS)
