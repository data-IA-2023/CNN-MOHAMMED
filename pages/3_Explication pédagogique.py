import streamlit as st

def main():
    st.title("Explication pédagogique")

    # Chemin vers le fichier HTML téléchargé
    chemin_fichier_html = "pages\CNN.html"  # Mettez à jour le chemin

    # Lecture du fichier HTML
    with open(chemin_fichier_html, 'r', encoding='utf-8') as f:
        contenu_html = f.read()

    # Affichage du contenu HTML
    st.markdown(contenu_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()