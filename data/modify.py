import numpy as np
import png
import sys

y_true = np.load('y_test_raw.npy')

for idx, array in enumerate(y_true):
    array = np.squeeze(array)
    array = np.interp(array, (array.min(), array.max()), (0, 256)).astype('uint8')

    png.from_array(array, 'L').save('y_true_{}.png'.format(idx))
    sys.exit()