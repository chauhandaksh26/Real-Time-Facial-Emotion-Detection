"""
Real-Time Facial Emotion Detection
Fixed for TensorFlow 2.x / Keras compatibility
"""

import numpy as np
import argparse
import cv2
import os
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="display", choices=["train", "display"])
    return ap.parse_args()

def build_model():
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    return model

def train():
    train_dir = 'data/train'
    val_dir   = 'data/test'

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical'
    )
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical'
    )

    model = build_model()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )
    model.summary()

    callbacks = [
        ModelCheckpoint('model.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]

    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=50,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=callbacks
    )

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['accuracy'],     label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Accuracy')
    axes[0].legend()

    axes[1].plot(history.history['loss'],     label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Loss')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('training_plot.png')
    plt.show()
    print("[INFO] Training complete. Model saved to model.h5")

def display():
    if not os.path.exists('model.h5'):
        print("[ERROR] model.h5 not found. Train the model first with --mode train")
        return

    model     = load_model('model.h5')
    face_haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap       = cv2.VideoCapture(0)

    EMOTION_LABELS = {
        0: 'Angry',
        1: 'Disgust',
        2: 'Fear',
        3: 'Happy',
        4: 'Neutral',
        5: 'Sad',
        6: 'Surprise'
    }

    print("[INFO] Webcam started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_haar.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype('float32') / 255.0
            roi = np.expand_dims(roi, axis=-1)
            roi = np.expand_dims(roi, axis=0)

            preds     = model.predict(roi, verbose=0)[0]
            label_idx = int(np.argmax(preds))
            label     = EMOTION_LABELS[label_idx]
            conf      = preds[label_idx] * 100

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.1f}%",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Facial Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = build_arg_parser()
    if args.mode == "train":
        train()
    elif args.mode == "display":
        display()
