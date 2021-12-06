import PhysThermalLibrary as ptl
import scipy.stats as ss
import matplotlib.pyplot as plt
import numpy as np

# import the simulation results - in counts not CR
sim_res = ptl.import_result('../data/PhysThermalMod2SIM.csv')
# import full normalised sim data and errors
sim_fn, _, _, sim_err = ptl.open_norm_and_errs(sim_res)
# import the dead time corrected experimental data in count rate
exp_res = ptl.import_result('../data/PhysThermalModEXP_CR_dtcorr.csv')
# normalise experimental results and get errors
exp_fn, _, _, exp_err = ptl.open_norm_and_errs(exp_res)
# correct the errors for 8*60 seconds of count rates
exp_err = exp_err/np.sqrt(8*60)
# reshape the data so regression can be performed
sim_fn_vals = sim_fn.values.reshape(12)
exp_fn_vals = exp_fn.values.reshape(12)
# perform the regression
result = ss.linregress(sim_fn_vals, exp_fn_vals)
# get the max value for x axis
x_max = np.max(sim_fn_vals)
# get x values for plotting fit
x_space = np.arange(0, 1.2 * x_max, (1.2 * x_max / 100))
# reshape simulated error counts into x values errors
x_err = ptl.get_full_norm_err_df(sim_res).values.reshape(12)
# reshape experimental errors into y errors
y_err = exp_err.values.reshape(12)
target_y = x_space * 1
regress_y = (x_space * result.slope) + result.intercept

ax = plt.gca()
ax.plot(x_space, target_y, color='black')
ax.plot(x_space, regress_y, color='#002768')
ax.errorbar(sim_fn_vals, exp_fn_vals, xerr=x_err*sim_fn_vals, yerr=y_err*exp_fn_vals, capsize=3, linestyle='none', color='navy')
ax.scatter(sim_fn_vals, exp_fn_vals, color='#002768', marker='.')
ax.set_ylim([0, np.max(target_y)])
ax.set_xlim([0, np.max(x_space)])
ax.set_xlabel('Simulated Counts')
ax.set_ylabel('Measured Counts')
plt.savefig('regress.png', dpi=600)
plt.show()
