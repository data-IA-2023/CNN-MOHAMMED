import streamlit as st
from PIL import Image
import os
import numpy as np
import mlflow.pyfunc
import matplotlib.pyplot as plt

# Chargement du modèle MLflow
logged_model = 'runs:/759d0ad6b1994ad7b8fb318ebde9762e/models'
# loaded_model = mlflow.pyfunc.load_model(logged_model)

@st.cache_resource  # Permet de mettre en cache le modèle chargé
def load_mlflow_model(logged_model):
    return mlflow.pyfunc.load_model(logged_model)

loaded_model = load_mlflow_model(logged_model)

# Chargement des classes (noms de dossiers) à partir du répertoire des données
data_dir = "animaux"
classes = os.listdir(data_dir)

# Fonction pour prétraiter une image
def preprocess_image(image, target_size):
    img = image.resize(target_size)
    img_array = np.array(img)
    # img_array = img_array / 255.0  # Normalisation
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Fonction pour effectuer une prédiction
def predict(image, model, target_size):
    preprocessed_image = preprocess_image(image, target_size)
    prediction = model.predict(preprocessed_image)
    return prediction

# Fonction pour convertir une image en JPEG
def convert_to_jpeg(input_image):
    # Convertir en mode RGB
    input_image = input_image.convert('RGB')
    # Créer un nom de fichier de sortie unique
    output_image_path = "output_image.jpg"
    # Enregistrer l'image en format JPEG
    input_image.save(output_image_path, 'JPEG')
    return output_image_path

# Créer une application Streamlit
st.title("Classification d'images")

# Ajouter une zone de téléchargement d'image
uploaded_file = st.file_uploader("Sélectionnez une image...")

# Si une image est téléchargée
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Image téléchargée', use_column_width=True)


    try:
        # Convertir l'image en JPEG
        jpeg_image_path = convert_to_jpeg(image)
        jpeg_image = Image.open(jpeg_image_path)
        # st.image(jpeg_image, caption='Image convertie en JPEG', use_column_width=True)

        # Effectuer la prédiction sur l'image convertie
        prediction = predict(jpeg_image, loaded_model, (180, 180))
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = classes[predicted_class_index]
        confidence = prediction[0][predicted_class_index]
        st.write(f"Classe prédite : {predicted_class_name}")
        st.write(f"Confiance : {confidence:.2f}")

        # Afficher l'histogramme des prédictions de classe
        class_counts = {class_name: prediction[0][i] for i, class_name in enumerate(classes)}
        plt.figure(figsize=(10, 6))
        plt.bar(class_counts.keys(), class_counts.values())
        plt.xlabel('Classe')
        plt.ylabel('Confiance de la prédiction')
        plt.title('Histogramme des prédictions de classe')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(plt)
    except Exception as e:
        st.write("Une erreur s'est produite lors de la prédiction :", e)
