import pandas as pd
import numpy as np
import sys
# load the indices that are null

thres = 20
y_true_app = []
y_pred_mse_app = []
y_pred_mae_app = []

y_true = np.load('data/y_true.npy')
y_pred_mse = np.load('data/y_pred_noNSE.npy')
y_pred_mae = np.load('data/y_pred_mae_full.npy')


for i in np.arange(0,939,1):

    y_true_binary = np.where(y_true[i]<26,0,1)

    y_pred_binary = np.where(y_pred_mse[i]<26,0,1)

    if np.sum(y_true_binary) >= thres and np.sum(y_pred_binary)>= thres:
        print(np.sum(y_true_binary))
        print(np.sum(y_pred_binary))
        y_true_app.append(y_true[i])
        y_pred_mse_app.append(y_pred_mse[i])
        y_pred_mae_app.append(y_pred_mae[i])

print(len(y_true_app))

y_true_app = np.asarray(y_true_app)
y_pred_mse_app = np.asarray(y_pred_mse_app)
y_pred_mae_app = np.asarray(y_pred_mae_app)

np.save('data/y_true_app.npy',y_true_app)
np.save('data/y_pred_mse_app.npy',y_pred_mse_app)
np.save('data/y_pred_mae_app.npy',y_pred_mae_app)
