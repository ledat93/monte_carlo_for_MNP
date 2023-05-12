import numpy as np
import model
import matplotlib.pyplot as plt
from tqdm import tqdm

# set parameters
box_size = (8,8,8)
num_batches = 1
origin_config = np.random.uniform(0, 2*np.pi, size=(num_batches,)+box_size+(2,))
# Ku = 9e3
# Ms = 466200.
# h = 15.9e3
Ku = 16e3
Ms = 281e3
h = 15.9e3
diameters = (7e-9)*np.ones(box_size)
distance = 10e-9
n_iters = int(1e5)
is_dipolar = False

param_obj = model.Parameters(Ku, Ms, h, box_size, distance, n_iters, is_dipolar=is_dipolar)
param_obj.setConfigAniso(0.0, 2.0*np.pi)
param_obj.setConfigExternalField(0.0, 0.0)
diameters = 7e-9*np.ones(box_size)
param_obj.setVolumes(diameters)
func = model.Funtions(param_obj.getParameters())
monter_carlo = model.MonteCarlo()

#print(param_obj.getParameters())


# run MC method for temperatures
min_temp, max_temp = 1, 100
temps = range(min_temp, max_temp)
num_temps = len(temps)
out_data = np.zeros((num_temps, 2))
x, y = [], []
idx = 0
for t in tqdm(temps):
    results = []
    for i in range(num_batches):
        config_i = origin_config[i]
        config = monter_carlo(t, config_i, param_obj)
        result_magnetic = func.computeMagnetization(config)
        results.append(result_magnetic)
    x.append(t)
    y.append(sum(results))

    out_data[idx, 0] = t
    out_data[idx, 1] = np.mean(results)
    idx += 1

# save file .npy
import os
root_path = './results'
file_name = 'data_n_1k4.npy'
file_path = os.path.join(root_path, file_name)
with open(file_path, 'wb') as fhandle:
    np.save(fhandle, out_data)

# plot results
enter_is_plot = input('Would you like to plot results?: ',)
avg_x = func.moveAverage(x)
avg_y = func.moveAverage(y)
if str(enter_is_plot) == 'yes' or str(enter_is_plot) == 'y':
    plt.scatter(x, y, color='blue')
    plt.plot(avg_x, avg_y, color='red')
    plt.show()
