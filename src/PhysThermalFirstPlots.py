import PhysThermalLibrary as tl
import matplotlib.pyplot as plt
if __name__ == "__main__":

    sim_res = tl.import_result('../data/PhysThermalMod2SIM.csv')
    sr_fn, sr_pn, sr_dn, sim_err = tl.open_norm_and_errs(sim_res)
    exp_res = tl.import_result('../data/PhysThermalModEXP_CR.csv')
    er_fn, er_pn, er_dn, exp_err = tl.open_norm_and_errs(exp_res)

    tl.plot_1df(sr_fn, sim_err, depth_func=True, transpose=True)
    plt.show()
    tl.plot_1df(er_fn, exp_err, exp=True, depth_func=True, transpose=True)
    plt.show()

    sr_dn_err = tl.get_depth_norm_err_df(sim_res)
    er_dn_err = tl.get_depth_norm_err_df(exp_res)
    tl.plot_vs_pos(sr_dn, er_dn, sr_dn_err, er_dn_err)
    plt.show()

    sr_pn_err = tl.get_pos_norm_err_df(sim_res)
    er_pn_err = tl.get_pos_norm_err_df(exp_res)
    tl.plot_vs_depth(sr_pn, er_pn, sr_pn_err, er_pn_err)
    plt.show()
