import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
from numpy import load

st.write("""
# Model Analytics
Choose a loss function.
""")

loss_functions = ['mean_squared_error', 'mean_absolute_error']

loss = st.radio("Pick a loss function", loss_functions)

y_test = load('data/y_test_raw.npy')
y_pred = load('data/y_pred_raw.npy')

number = st.slider("Pick a sample number (0-939)",0,939)

y_true = np.squeeze(y_test[number])

st.image(y_true)

