import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

# Ruta al dataset
dataset_dir = "Dataset/Fossil/Geo Fossils-I Dataset"

# Parámetros del dataset
img_size = (150, 150)
batch_size = 32

# Cargar dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

# Guardamos las clases antes del prefetch
class_names = train_ds.class_names
num_classes = len(class_names)

# Cache y prefetch para performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Modelo CNN simple
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(150, 150, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compilar modelo
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenar el modelo
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# === GUARDAR EL MODELO Y LAS CLASES ===
# 1. Guardar el modelo en formato moderno (.keras)
model.save("fossil_classifier.keras")
print(" Modelo guardado como 'fossil_classifier.keras'")

# 2. Guardar las clases en un archivo .npy (útil para cargar después)
np.save("class_names.npy", np.array(class_names))
print(" Clases guardadas como 'class_names.npy'")