import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def open_norm_and_errs(res_df):
    err_df = compute_errs(res_df)
    fn_df = full_norm(res_df)
    pn_df = pos_norm(res_df)
    dn_df = depth_norm(res_df)

    return fn_df, pn_df, dn_df, err_df


def import_result(filename):
    result = pd.read_csv(filename, header=[0])
    result.set_index('Depth', inplace=True)

    return result


def full_norm(df):
    n_df = df / df.sum().sum()
    return n_df


def pos_norm(df):
    n_df = df / df.sum()
    return n_df


def depth_norm(df):
    n_df = df.transpose() / df.sum(axis=1)
    return n_df


def plot_vs_pos(mod_df, exp_df, m_err=None, e_err=None):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharey=True)
    mod_mark, mod_label, mod_clr = get_plot_sets(False)
    exp_mark, exp_label, exp_clr = get_plot_sets(True)
    keys, x_label = set_depth_or_pos(True)
    # if m_err is not None:
    #     m_err = pos_transpose(m_err, True)
    # if e_err is not None:
    #     e_err = pos_transpose(e_err, True)
    count = 0
    for i in range(2):
        for j in range(2):

            axes[i][j].set_title('%s cm Depth' % keys[count])
            axes[i][j].scatter(mod_df.index, mod_df[keys[count]].values, marker=mod_mark, color=mod_clr[count], s=10)
            axes[i][j].scatter(exp_df.index, exp_df[keys[count]].values, marker=exp_mark, color=exp_clr[count], s=10)
            if m_err is not None:
                axes[i][j].errorbar(mod_df.index, mod_df[keys[count]].values,
                                    yerr=m_err[keys[count]].values * mod_df[keys[count]].values,
                                    color=mod_clr[count], linestyle='none', capsize=3)
            if e_err is not None:
                axes[i][j].errorbar(exp_df.index, exp_df[keys[count]].values,
                                    yerr=e_err[keys[count]].values * exp_df[keys[count]].values,
                                    color=exp_clr[count], linestyle='none', capsize=3)
            axes[i][j].legend(['Model', 'Experiment'], loc=4)
            axes[i][j].set_xlabel(x_label)
            axes[i][j].set_ylabel('Normalised Counts')
            count += 1

    plt.show()
    return


def get_plot_sets(exp):
    if exp:
        mrk = 'o'
        lbl = 'Experiment'
        colour_lst = ['#00276B', '#0046BF', '#1F6AED', '#8BB0F0']

    else:
        mrk = 's'
        lbl = 'Model'
        colour_lst = ['#00276B', '#0046BF', '#1F6AED', '#8BB0F0']
    return mrk, lbl, colour_lst


def set_depth_or_pos(pos):
    if pos:
        keys = [65.0, 67.5, 70.0, 72.5]
        x_lbl = 'Position'
    else:
        keys = ['P1', 'P2', 'P3']
        x_lbl = 'Depth'
    return keys, x_lbl


def pos_transpose(df, pos):
    if pos:
        df = df.transpose()
    return df


def plot_vs_depth(mod_df, exp_df, m_err=None, e_err=None):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    mod_mark, mod_label, mod_clr = get_plot_sets(False)
    exp_mark, exp_label, exp_clr = get_plot_sets(True)
    keys, x_label = set_depth_or_pos(False)

    for i in range(len(keys)):
        axes[i].set_title(keys[i])
        axes[i].scatter(mod_df.index, mod_df[keys[i]].values, marker=mod_mark, color=mod_clr[i], s=10)
        if m_err is not None:
            axes[i].errorbar(mod_df.index, mod_df[keys[i]].values, yerr=m_err[keys[i]].values * mod_df[keys[i]].values,
                             color=mod_clr[i], linestyle='none', capsize=3)
        if e_err is not None:
            axes[i].errorbar(exp_df.index, exp_df[keys[i]].values,
                             yerr=e_err[keys[i]].values * exp_df[keys[i]].values,
                             color=exp_clr[i], linestyle='none', capsize=3)
        axes[i].scatter(exp_df.index, exp_df[keys[i]].values, marker=exp_mark, color=exp_clr[i], s=10)
        axes[i].legend(['Model', 'Experiment'])
        axes[i].set_xlabel(x_label)
        axes[i].set_ylabel('Normalised Counts')

    plt.show()

    return


def plot_1df(df, offset, err_df=None, exp=False, depth_func=False, transpose=False):
    mark, lab, colour_list = get_plot_sets(exp)
    keys, x_label = set_depth_or_pos(depth_func)

    if transpose:
        df = pos_transpose(df, depth_func)
    if err_df is not None:
        err_df = pos_transpose(err_df, depth_func)
    ax = plt.gca()
    for i in range(len(keys)):
        ax.scatter(df.index + offset[keys[i]], df[keys[i]].values, marker=mark, s=10, color=colour_list[i],
                   label="%s %s" % (keys[i], lab))
        if err_df is not None:
            ax.errorbar(df.index + offset[keys[i]], df[keys[i]].values, yerr=err_df[keys[i]].values * df[keys[i]].values,
                        color=colour_list[i], linestyle='none', capsize=3)
    ax.legend()
    ax.set_xlabel(x_label)
    ax.set_ylabel('Normalised Counts')
    return ax


def get_full_norm_err_df(df):
    full_norm_const = df.sum().sum()
    fn_err_df = np.sqrt((1/df) + (1/full_norm_const))
    return fn_err_df


def get_pos_norm_err_df(df):
    pos_norm_consts = df.sum()
    pn_err_df = np.sqrt((1/df) + (1/pos_norm_consts))
    return pn_err_df


def get_depth_norm_err_df(df):
    depth_norm_consts = df.transpose().sum()
    dn_err_df = np.sqrt((1/df.transpose()) + (1/depth_norm_consts))

    return dn_err_df


def get_residuals_error(df1, df1_err, df2, df2_err):
    res_err_df = np.sqrt(np.square(df1 * df1_err) + np.square(df2*df2_err))
    return res_err_df


def compute_errs(df):
    return 1/np.sqrt(df)

