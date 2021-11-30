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


number = st.number_input("Pick a sample number (0-939)",0,939)

y_true = np.squeeze(y_test[number])
y_true = np.interp(y_true, (y_true.min(), y_true.max()), (0, +255)).astype('uint8')
y_true[y_true < 0.3] = 0

y_pred = np.squeeze(y_pred[number])
y_pred = np.interp(y_pred, (y_pred.min(), y_pred.max()), (0, +1))
y_pred[y_pred < 0.3] = 0

# f, axs = plt.subplots(1,2,figsize=(15,15))

# plt.subplot(121)
# ax = plt.gca()
# cs = plt.contourf(y_true)

# plt.subplot(122)
# ax = plt.gca()
# cs = plt.contourf(y_pred)

# st.pyplot(f)

palette = [255,0,0,    # 0=red
           0,255,0,    # 1=green
           0,0,255,    # 2=blue
           255,255,0,  # 3=yellow
           0,255,255]  # 4=cyan
# Pad with zeroes to 768 values, i.e. 256 RGB colours
palette = palette + [0]*(768-len(palette))

# Convert Numpy array to palette image
pi = Image.fromarray(y_true,'P')

# Put the palette in
pi.putpalette(palette)

# Display and save
pi.show()
pi.save('result.png')

# st.image(y_true,width=300)
# st.image(y_pred,width=300)

