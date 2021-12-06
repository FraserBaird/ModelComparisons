import numpy as np

x = np.arange(0, 250, 0.01)
y = (x*0.0172) / (np.exp(0.0172 * x * 0.000687) + 0.0172 - 1)

target_y = 100

y_diff = np.abs(y - target_y)
target_ind = np.where(y_diff == np.min(y_diff))
print('Corrected value: %s' % x[target_ind])