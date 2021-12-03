import pandas as pd
import numpy as np
import sys
# load the indices that are null

thres = 20
y_true_app = []
y_pred_mse_app = []
y_pred_mae_app = []

y_true = np.load('data/y_true.npy')
y_pred_mse = np.load('data/y_pred_MSE_best.npy')
y_pred_mae = np.load('data/y_pred_mae_app.npy')

# print(y_pred_mae.shape)
# sys.exit()


for i in np.arange(0,939,1):
    y_true_binary = np.where(y_true[i]<26,0,1)
    if np.sum(y_true_binary) >= thres:
        print(np.sum(y_true_binary))
        y_true_app.append(y_true[i])
        y_pred_mse_app.append(y_pred_mse[i])
        y_pred_mae_app.append(y_pred_mae[i])

print(len(y_true_app))

y_true_app = np.asarray(y_true_app)
y_pred_mse = np.asarray(y_pred_mse)
y_pred_mae = np.asarray(y_pred_mae)

np.save('data/y_true_app.npy',y_true_app)
np.save('data/y_pred_mse_app.npy',y_pred_mse_app)
np.save('data/y_pred_mae_app.npy',y_pred_mae_app)
