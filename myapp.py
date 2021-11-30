import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
from numpy import load

st.write("""
# Model Analytics
Choose a loss function.
""")

loss_functions = ['mse', 'mae']

loss = st.radio("Pick a loss function", loss_functions)

y_test = load('data/y_test_raw.npy')

if loss == 'mse':
    y_pred = load('data/y_pred_raw.npy')
else:
    y_pred = load('data/y_pred_{}.npy'.format(loss))

number = st.slider("Pick a sample number (0-939)",0,939)

y_true = np.squeeze(y_test[number])
y_true = np.interp(y_true, (y_true.min(), y_true.max()), (0, +1))
y_true[y_true < 0.2] = 0

y_pred = np.squeeze(y_pred[number])
y_pred = np.interp(y_pred, (y_pred.min(), y_pred.max()), (0, +1))
y_pred[y_pred < 0.2] = 0

# f, axs = plt.subplots(1,2,figsize=(15,15))

# plt.subplot(121)
# ax = plt.gca()
# cs = plt.contourf(y_true)

# plt.subplot(122)
# ax = plt.gca()
# cs = plt.contourf(y_pred)

# st.pyplot(f)

st.image(y_true,width=300)
st.image(y_pred,width=300)

