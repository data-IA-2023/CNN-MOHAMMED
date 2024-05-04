import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

# Charger le modèle sauvegardé
@st.cache_resource  # Permet de mettre en cache le modèle chargé
def load_model(file):
    return keras.models.load_model(file)

model = load_model(file= "modeles\model_catvsdog.h5")
image_size = (180, 180)

# Définir une fonction pour faire des prédictions
def predict_img_array(img_array):
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    score = float(predictions[0][0])
    return score

# Définir l'interface de l'application
def main():
    st.title("Classificateur de Chat ou Chien")
    uploaded_file = st.file_uploader("Sélectionnez une image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = image.resize(image_size)
        st.image(image, caption="Image chargée", use_column_width=True)
        
        img_array = np.array(image) # Normaliser les valeurs des pixels
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        score = predict_img_array(img_tensor)

        if score >= 0.5:
            prediction_label = "Chien"
        else:
            prediction_label = "Chat"
        
        st.write(f"Cette image est prédite comme {prediction_label} avec une confiance de {100 * abs(score - 0.5) * 2:.2f}%")

if __name__ == "__main__":
    main()


