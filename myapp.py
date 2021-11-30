import streamlit as st
import pandas as pd
from PIL import Image
from numpy import load

st.write("""
# Model Analytics
Choose a loss function.
""")

loss_functions = ['mean_squared_error', 'mean_absolute_error']

loss = st.radio("Pick a loss function", loss_functions)

y_test = load('/data/y_test_raw.npy')
y_pred = load('/data/y_pred_raw.npy')

st.text_input("Pick a sample number (0-939)", key="sample")

sample = int(st.session_state.sample)


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(uploaded_file, caption = 'Input image', use_column_width=False)
