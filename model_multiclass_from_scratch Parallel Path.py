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

# Initialiser MLflow pour afficher le tableau de bord : mlflow ui
mlflow.set_experiment("multiclass_model_from_scratch")

# Activer l'autologging de MLflow pour TensorFlow
mlflow.tensorflow.autolog()

# Charger et prétraiter les données
image_size = (180, 180)
batch_size = 128  # Peut être réduit en fonction de la mémoire GPU disponible

# Obtenir les noms des dossiers (classes) dans le répertoire des données
data_dir = "animaux"
classes = os.listdir(data_dir)
num_classes = len(classes)  # Nombre total de classes d'animaux que vous avez
print('nombre de classes=', num_classes)

prefetch_buffer_size = tf.data.experimental.AUTOTUNE

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

# Augmentation des données
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

# model = keras.Sequential([
#     layers.Rescaling(1/255, input_shape=(180, 180, 3)),
    
#     layers.Conv2D(32, 3, padding='same', activation='relu'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D(),
    
#     layers.Conv2D(64, 3, padding='same', activation='relu'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D(),
    
#     layers.Conv2D(128, 3, padding='same', activation='relu'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D(),
    
#     layers.Conv2D(256, 3, padding='same', activation='relu'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D(),
    
#     layers.Flatten(),
#     layers.Dense(512, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(num_classes, activation='softmax')
# ])


# def make_multiclass_model(input_shape, num_classes):
#     inputs = keras.Input(shape=input_shape)

#     # Réduction de l'échelle des valeurs de pixel
#     scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
#     x = scale_layer(inputs)

#     x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.MaxPooling2D()(x)

#     previous_block_activation = x

#     for size in [64, 128, 256]:
#         x = layers.Activation("relu")(x)
#         x = layers.SeparableConv2D(size, 3, padding="same")(x)
#         x = layers.BatchNormalization()(x)

#         x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

#         residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
#         previous_block_activation = layers.MaxPooling2D(3, strides=2, padding="same")(previous_block_activation)
#         x = layers.add([x, residual])

#     x = layers.Flatten()(x)
#     x = layers.Dense(512, activation='relu')(x)
#     x = layers.Dropout(0.5)(x)
#     outputs = layers.Dense(num_classes, activation='softmax')(x)

#     return keras.Model(inputs, outputs)




def residual_block(x, filters, kernel_size=3):
    y = layers.Conv2D(filters, kernel_size, padding='same')(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = layers.Conv2D(filters, kernel_size, padding='same')(y)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = layers.Add()([x, y])
    return layers.ReLU()(y)

def make_multiclass_model(input_shape, num_classes):
    input_img = layers.Input(shape=input_shape)

    scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
    x = scale_layer(input_img)

    # First Parallel Path
    path1 = layers.Conv2D(16, 3, padding='same')(x)
    path1 = layers.BatchNormalization()(path1)
    path1 = layers.ReLU()(path1)
    path1 = layers.MaxPooling2D()(path1)

    path1 = residual_block(path1, 16)
    path1 = layers.MaxPooling2D()(path1)

    # Second Parallel Path
    path2 = layers.Conv2D(32, 3, padding='same')(x)
    path2 = layers.BatchNormalization()(path2)
    path2 = layers.ReLU()(path2)
    path2 = layers.MaxPooling2D()(path2)

    path2 = residual_block(path2, 32)
    path2 = layers.MaxPooling2D()(path2)

    # Merge paths
    merged = layers.Concatenate()([path1, path2])

    # Continue with your network architecture
    merged = layers.Conv2D(64, 3, padding='same', activation='relu')(merged)
    merged = layers.BatchNormalization()(merged)
    merged = layers.MaxPooling2D()(merged)

    merged = layers.Conv2D(128, 3, padding='same', activation='relu')(merged)
    merged = layers.BatchNormalization()(merged)
    merged = layers.MaxPooling2D()(merged)

    merged = layers.Flatten()(merged)
    merged = layers.Dense(256, activation='relu')(merged)
    merged = layers.Dropout(0.5)(merged)
    output = layers.Dense(num_classes, activation='softmax')(merged)

    model = keras.Model(input_img, output)

    return model

model=make_multiclass_model(input_shape=(180, 180, 3),num_classes=num_classes)

# Compilation du modèle
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# Entraîner le modèle
epochs = 25
print("Fitting the model")

# Entraîner le modèle avec MLflow activé
with mlflow.start_run() as run:
    # Entraînement du modèle
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)

    # # Enregistrement des métriques système à chaque point de l'expérience
    # for epoch in range(epochs):
    #     # Obtenir les métriques système
    #     cpu_percent = psutil.cpu_percent()
    #     memory_percent = psutil.virtual_memory().percent

    #     # Enregistrer les métriques système
    #     mlflow.log_metric("cpu_percent", cpu_percent, step=epoch)
    #     mlflow.log_metric("memory_percent", memory_percent, step=epoch)

    # Évaluation du modèle
    val_loss, val_accuracy = model.evaluate(val_ds)
    print("Validation accuracy:", val_accuracy)
    print("Validation loss:", val_loss)

    # Enregistrer le modèle
    model.save("multiclass_model_from_scratch.h5")

    # Save the model using MLflow
    mlflow.keras.log_model(model, "models")

    # Enregistrer les courbes d'apprentissage dans un fichier
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plotting the training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['sparse_categorical_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_sparse_categorical_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.show()

    # Sauvegarder les courbes d'apprentissage dans un fichier
    plt.savefig('training_curves_from_scratch.png')

mlflow.end_run()
