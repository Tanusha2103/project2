#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AER850
Project 2
Steps 1-4

Tanusha Lingam
501130352

"""

import os, json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TerminateOnNaN


# ---------------- Step 1 - Data Processing ----------------

IMG_SIZE = (500, 500)
INPUT_SHAPE = (500, 500, 3)
BATCH_SIZE = 32

DATA_PATH = "/Users/tanu/Desktop/AER850-Projects/project 2/Data"

# train data relative path
TRAIN_PATH = os.path.join(DATA_PATH, "train")

# test data relative path
TEST_PATH = os.path.join(DATA_PATH, "test")  # used for Step 5

# validation data relative path
VALID_PATH = os.path.join(DATA_PATH, "valid")

# data augmentation
TRAIN_DATA = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    rotation_range=10,
    horizontal_flip=True,
    fill_mode="nearest",
    )
VALID_DATA = ImageDataGenerator(rescale=1./255)

# train and validation generator
train_generator = TRAIN_DATA.flow_from_directory ( TRAIN_PATH,                                        
        target_size=IMG_SIZE, 
        batch_size=BATCH_SIZE, 
        class_mode="categorical"
        )
valid_generator = VALID_DATA.flow_from_directory ( VALID_PATH,                                           
        target_size=IMG_SIZE, 
        batch_size=BATCH_SIZE, 
        class_mode="categorical",
        shuffle=False
        )

class_map = {v: k for k, v in train_generator.class_indices.items()}
with open("/Users/tanu/Desktop/AER850-Projects/project 2/models/classes.json", "w") as f:
    json.dump(class_map, f, indent=2)
    
# ---------------- Step 2 - Neural Network Architecture Design ----------------

model = models.Sequential([
    layers.Input(shape=INPUT_SHAPE),

    # Block 1
    layers.Conv2D(32, kernel_size=3, strides=1, padding="same", activation="relu"),
    layers.MaxPooling2D(pool_size=2),

    # Block 2
    layers.Conv2D(64, kernel_size=3, strides=1, padding="same", activation="relu"),
    layers.MaxPooling2D(pool_size=2),

    # Block 3
    layers.Conv2D(128, kernel_size=3, strides=1, padding="same", activation="relu"),
    layers.MaxPooling2D(pool_size=2),

    # Block 4
    layers.Conv2D(256, kernel_size=3, strides=1, padding="same", activation="relu"),
    layers.MaxPooling2D(pool_size=2),

    # Flatten, Dropout & Dense
    layers.Flatten(),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(3, activation="softmax"),
])

# ---------------- Step 3 - Hyperparameter Analysis ----------------

adam_optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)

model.compile(
    optimizer=adam_optimizer,
    loss='categorical_crossentropy',   
    metrics=['accuracy']              
)

model.summary()

# ---------------- Step 4 - Model Evaluation ----------------

es = EarlyStopping(
    monitor="val_loss", 
    patience=25, 
    restore_best_weights=True
    )
ck = ModelCheckpoint(
    "models/custom_cnn.keras", 
    save_best_only=True, 
    monitor="val_accuracy", 
    mode="max"
    )

nan_guard = TerminateOnNaN()

# Train the model and record history
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=20,
    verbose=1,
    callbacks=[es, ck, nan_guard]
)

# Plot training vs validation accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_plot.png', dpi=150)

# Plot training vs validation loss
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_plot.png', dpi=150)
 
plt.show()



