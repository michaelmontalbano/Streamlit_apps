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

y_test = load('data/y_true_app.npy')

if loss == 'mse':
    y_pred = load('data/y_pred_mse_app.npy')
else:
    y_pred = load('data/y_pred_{}_app.npy'.format(loss))


number = st.number_input("Pick a sample number (0-939)",0,939)

y_true = np.squeeze(y_test[number])
y_pred = np.squeeze(y_pred[number])

f, axs = plt.subplots(1,2,figsize=(16,8))

plt.subplot(121)
ax = plt.gca()
cs = plt.contourf(y_true,levels=MESH_bounds,colors=MESH_colors, extend='both',orientation='horizontal', shrink=0.5, spacing='proportional')   
plt.colorbar(cs, ticks=MESH_bounds)
f.tight_layout(pad=3.0)
plt.ylabel('y (1/2 km)')
plt.xlabel('x (1/2 km)')
plt.xlim([0,60])
plt.xticks([0,10,20,30,40,50,60])
plt.yticks([0,10,20,30,40,50,60])
plt.title('True MESH  #{} (mm)'.format(number))

plt.subplot(122)
ax = plt.gca()
cs = plt.contourf(y_pred,levels=MESH_bounds,colors=MESH_colors, extend='both',orientation='horizontal', shrink=0.5, spacing='proportional')   
plt.colorbar(cs, ticks=MESH_bounds)
f.tight_layout(pad=3.0)
plt.ylabel('y (1/2 km)')
plt.xlabel('x (1/2 km)')
plt.xlim([0,60])
plt.xticks([0,10,20,30,40,50,60])
plt.yticks([0,10,20,30,40,50,60])
plt.title('Predicted MESH with {} #{} (mm)'.format(loss,number))

st.pyplot(f)

ax = plt.gca()
bounds = MESH_bounds

