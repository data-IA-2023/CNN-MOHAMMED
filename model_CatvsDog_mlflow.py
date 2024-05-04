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

# Initialiser MLflow pour afficher le tableau de bord : mlflow ui
mlflow.set_experiment("Cat_vs_Dog_Classifier")

# Activer l'autologging de MLflow pour TensorFlow
mlflow.tensorflow.autolog()

# Charger et prétraiter les données
image_size = (180, 180)
batch_size = 64  # Peut être réduit en fonction de la mémoire GPU disponible
prefetch_buffer_size = tf.data.experimental.AUTOTUNE

train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="both",
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

def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

# Utilisation d'un modèle pré-entraîné
base_model = keras.applications.Xception(
    weights="imagenet",  # Utiliser les poids pré-entraînés sur ImageNet
    input_shape=image_size + (3,),
    include_top=False,
)
base_model.trainable = False  # Freezing the base model

# Création du modèle en ajoutant un classifieur au-dessus du modèle pré-entraîné
inputs = keras.Input(shape=image_size + (3,))

x = data_augmentation(inputs)
scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
x = scale_layer(inputs)

x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(1, activation="sigmoid")(x)##########################################
model = keras.Model(inputs, outputs)

model.summary(show_trainable=True)

# Compilation du modèle
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

# Entraîner le modèle
epochs = 25
print("Fitting the top layer of the model")

# Entraîner le modèle avec MLflow activé
with mlflow.start_run() as run:
    # Entraînement du modèle
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)

    
        # Enregistrement des métriques système à chaque point de l'expérience
    for epoch in range(epochs):
        # Obtenir les métriques système
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent

        # Obtenir les métriques GPU
        gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.total,memory.used", "--format=csv,nounits,noheader"])
        gpu_info = gpu_info.decode("utf-8")
        gpu_info = gpu_info.split("\n")
        gpu_utilization, gpu_total_memory, gpu_used_memory = map(float, gpu_info[0].split(","))
        gpu_memory_percent = gpu_used_memory / gpu_total_memory * 100

        # Enregistrer les métriques système et GPU
        mlflow.log_metric("cpu_percent", cpu_percent, step=epoch)
        mlflow.log_metric("memory_percent", memory_percent, step=epoch)
        mlflow.log_metric("gpu_utilization", gpu_utilization, step=epoch)
        mlflow.log_metric("gpu_memory_percent", gpu_memory_percent, step=epoch)
    
    
    # Évaluation du modèle
    loss, accuracy = model.evaluate(val_ds)
    print("Validation accuracy:", accuracy)
    print("Validation loss:", loss)
    
    # Sauvegarder le modèle dans MLflow
    mlflow.keras.log_model(model, "models")
    
    # Enregistrer le modèle
    model.save("my_model.h5")
    
    
    
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
    plt.plot(history.history['binary_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_binary_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.show()

    # Sauvegarder les courbes d'apprentissage dans un fichier
    plt.savefig('training_curves.png')


mlflow.end_run()
#TEST

model.summary()
image_size = (180, 180)
# Load and display the image
img = keras.utils.load_img("archive\googledog.jpg", target_size=image_size)
plt.imshow(img)

# Convert the image to an array and add a batch axis
img_array = keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Make predictions using the model
predictions = model.predict(img_array)
# score = float(keras.backend.sigmoid(predictions[0][0]))  # Assuming you're using TensorFlow backend
score = float(predictions[0][0]) 
print(f"Cette image est à {100 * (1 - score):.2f}% un chat et à {100 * score:.2f}% un chien.")


# Print the prediction results
if score >= 0.5:
    prediction_label = "Dog"
else:
    prediction_label = "Cat"
    
plt.title(f"Cette image est prédite comme {prediction_label} avec une confiance de {100 * abs(score - 0.5) * 2:.2f}%")
plt.show()




