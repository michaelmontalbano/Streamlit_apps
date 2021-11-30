import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
from numpy import load
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.colors import rgb2hex

MESH_colors = ['#aaaaaa','#00ffff','#0080ff','#0000ff','#007f00','#00bf00','#00ff00','#ffff00','#bfbf00','#ff9900','#ff0000','#bf0000','#7f0000','#ff1fff']
MESH_bounds = [9.525,15.875,22.225,28.575,34.925,41.275,47.625,53.975,60.325,65,70,75,80,85]

# scalers = open_pickle('scaler_raw.pkl')
# scaler = scalers[0]


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


number = st.number_input("Pick a sample number (0-939)",0,939)

y_true = np.squeeze(y_test[number])
#y_true = np.interp(y_true, (y_true.min(), y_true.max()), (0, +255)).astype('uint8')
#y_true[y_true < 0.3] = 0

y_pred = np.squeeze(y_pred[number])
#y_pred = np.interp(y_pred, (y_pred.min(), y_pred.max()), (0, +1))
#y_pred[y_pred < 0.3] = 0

# f, axs = plt.subplots(1,2,figsize=(15,15))

# plt.subplot(121)
# ax = plt.gca()
# cs = plt.contourf(y_true)

# plt.subplot(122)
# ax = plt.gca()
# cs = plt.contourf(y_pred)

# st.pyplot(f)

ax = plt.gca()
bounds = MESH_bounds

