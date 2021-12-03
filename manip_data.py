import pandas as pd
import numpy as np

# load the indices that are null
nulls = pd.read_csv('null_indices_mse_15.csv')

y_true_app = []
y_pred_mse = []
y_pred_mae = []

y_true = np.load('data/y_true_best.npy')
y_pred_mse = np.load('data/y_pred_best.npy')
y_pred_mae = np.load('data/y_pred_mae.npy')

for i in np.arange(0,939,1):
    if i not in nulls:
        y_true_app.append(y_true[i])
        y_pred_mse.append(y_pred_mse[i])
        y_pred_mae.append(y_pred_mae[i])
np.save('data/y_true_app.npy',y_true_app)
np.save('data/y_pred_mse.npy',y_pred_mse)
np.save('data/y_pred_mae.npy',y_pred_mae)
