import streamlit as st

def main():
    # Centralisation de l'image
    st.image("logo.png", width=200, use_column_width=True, output_format="auto")

    # Centralisation du titre
    st.markdown("<h1 style='text-align: center;'>Classification d'images - Deep Learning & CNN</h1>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()