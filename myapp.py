import streamlit as st
import pandas as pd
import PIL

st.write("""
# My first application
Choose an image.
""")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(uploaded_file, caption = 'Input image', use_column_width=True)
