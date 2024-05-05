import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from PIL import Image
import matplotlib.pyplot as plt
import mlflow
from mlflow import keras as mlflow_keras
import psutil
import subprocess

# Set up GPU memory limit
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])  # 4096 MB = 4 GB
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Initialize MLflow
mlflow.set_experiment("Animal_Classifier")
mlflow.tensorflow.autolog()

# Load and preprocess data
# image_size = (180, 180)#########################################
image_size = (380, 380)
batch_size = 64
data_dir = "animaux"
classes = os.listdir(data_dir)
num_classes = len(classes)  # Nombre total de classes d'animaux que vous avez
print('nombre de classes=', num_classes)



train_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

val_ds = keras.utils.image_dataset_from_directory(
    "animaux",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

# Data augmentation
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
]

def data_augmentation(images, labels):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images, labels

train_ds = train_ds.map(data_augmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Le préchargement des échantillons dans la mémoire GPU aide à maximiser l'utilisation du GPU.
train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE)

# Use a pre-trained model
# base_model = keras.applications.EfficientNetB3(######################################
# base_model = keras.applications.EfficientNetB7(
base_model = keras.applications.EfficientNetB4(
    weights="imagenet",
    input_shape=image_size + (3,),
    include_top=False,
)
base_model.trainable = False

# Create the model
inputs = keras.Input(shape=image_size + (3,))
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = keras.Model(inputs, outputs)


# Affiche un diagramme du modèle montrant la connexion entre les différentes couches, avec les formes de sortie de chaque couche
# keras.utils.plot_model(model, show_shapes=True)
model.summary()

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# Train the model
epochs = 25
print("Fitting the top layer of the model")

# Start MLflow run
with mlflow.start_run() as run:
    # Training step
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)


    # Evaluation
    val_loss, val_accuracy = model.evaluate(val_ds)
    print("Validation accuracy:", val_accuracy)
    print("Validation loss:", val_loss)

    # Save the model
    model.save("multiclass_model.h5")
    
    # Log the model using MLflow
    mlflow.keras.log_model(model, "models")
    
    # Plotting training curves
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['sparse_categorical_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_sparse_categorical_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.show()

    plt.savefig('training_curves.png')

# End MLflow run
mlflow.end_run()
