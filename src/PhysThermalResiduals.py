import PhysThermalLibrary as ptl
import matplotlib.pyplot as plt
import numpy as np

# import the simulation results - in counts not CR
sim_res = ptl.import_result('../data/PhysThermalMod2SIM.csv')
# get the various normalised results and fractional error for the simulated results
sim_fn, sim_pn, sim_dn, sim_err = ptl.open_norm_and_errs(sim_res)
# import the deadtime corrected experimental results - in count rate by necessity
exp_res = ptl.import_result('../data/PhysThermalModEXP_CR_dtcorr.csv')
# normalise the experimental results
exp_fn, exp_pn, exp_dn, _ = ptl.open_norm_and_errs(exp_res)

# compute the experimental error in position normalised data - have to include factor of 8*60 to account for the count
# rate being used instead of counts
exp_pn_err = ptl.get_pos_norm_err_df(exp_res * 8 * 60)
# compute simulated position normalised error
sim_pn_err = ptl.get_pos_norm_err_df(sim_res)
# compute residuals for position normalised data
pn_residuals = (sim_pn - exp_pn)
# compute the fractional error of the positional normalised residuals
pn_res_err_frac = ptl.get_residuals_error(sim_pn, sim_pn_err, exp_pn, exp_pn_err)/pn_residuals
pn_offsets = {'P1': - 0.15, 'P2': 0, 'P3': 0.15}

sim_dn_err = ptl.get_depth_norm_err_df(sim_res)
exp_dn_err = ptl.get_depth_norm_err_df(exp_res * 8 * 60)
dn_residuals = sim_dn - exp_dn
dn_res_err_frac = ptl.get_residuals_error(sim_dn, sim_dn_err, exp_dn, exp_dn_err)/dn_residuals

ax = ptl.plot_1df(pn_residuals, pn_offsets, err_df=pn_res_err_frac, depth_func=False)

ax.axhline(0, color='black')
ax.legend(['_', 'P1 Residuals', 'P2 Residuals', 'P3 Residuals'])
ax.set_ylabel('Residual')
plt.show()
# ax2 = ptl.plot_1df(dn_residuals, err_df=dn_res_err_frac.transpose(), depth_func=True)
# ax2.axhline(0, color='black')
# ax2.set_ylabel('Residual')
# ax2.legend(['_', '65cm Residuals', '67.5cm Residuals', '70cm Residuals', '72.5cm Residuals'])
# plt.show()
