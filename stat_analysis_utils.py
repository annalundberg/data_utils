
import numpy as np
import pandas as pd
import os
from scipy import stats
from matplotlib import pyplot as plt


def fit_line(df, col_a, col_b):
    '''DESC:
    Best fit line using pearsons R.
    INPUTS:
    df: Pandas DataFrame, contains data to fit line
    col_a: str, column name from df, set1 of values
    col_b: str, column name from df, set2 of values
    OUTPUTS:
    m: float, slope of line fit
    c: float, y-intercept of line fit
    p: float, Pearson's correlation of fit
    '''
    r,p = stats.pearsonr(df[col_a], df[col_b])
    # Gradient of line of best-fit
    m = r * np.std(df[col_b]) / np.std(df[col_a])
    # Y-intercept of line of best-fit
    c = np.mean(df[col_b]) - m * np.mean(df[col_a])
    return m, c, p


def Lins_CCC(df, col_true, col_pred):
    """DESC:
    Lin's Concordance correlation coefficient.
    INPUTS:
    df: Pandas DataFrame, contains data to fit line
    col_true: str, column name from df, reference set of values
    col_pred: str, column name from df, test set of values
    OUTPUTS:
    ccc: float, Lin's concordance correlation coefficient between 2 value sets
    """
    # Remove NaNs
    df = df.dropna()
    y_true = df[col_true]
    y_pred = df[col_pred]
    # Pearson product-moment correlation coefficients
    cor = np.corrcoef(y_true, y_pred)[0][1]
    # Mean
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    # Variance
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    # Standard deviation
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    ccc = numerator / denominator
    return ccc


def plot_ccc(df, col_true, col_pred, outdir=None, opt_detail=None):
    """DESC:
    Plots paired value sets, with line fit via Pearson's. Reports on
    plot Lin's CCC and fit line equation.
    INPUTS:
    df: Pandas DataFrame, contains data to fit line
    col_true: str, column name from df, reference set of values
    col_pred: str, column name from df, test set of values
    outdir: str, dir path to save plot in, default None does not save plot
    opt_detail: str, detail to include in plot naming, None does not alter name
    OUTPUTS:
    displays plot and will save if outdir given
    """
    if opt_detail is not None:
        plot_name = 'CCC {} {} vs {}'.format(opt_detail,col_true,col_pred)
    else:
        plot_name = 'CCC {} vs {}'.format(col_true,col_pred)
    # Plot
    ax = plt.axes()
    ax.set(title=plot_name, xlabel=col_true, ylabel=col_pred)
    # Scatter plot
    ax.scatter(df[col_true], df[col_pred], c='k', s=20, alpha=0.6, marker='o')
    # Get axis limits
    left, right = plt.xlim()
    # Keep current axis limits
    ax.set_xlim(left, right)
    # Line of best-fit
    x = np.array([left, right])
    m,c,_ = fit_line(df, col_true, col_pred)
    lccc = Lins_CCC(df, col_true, col_pred)
    y = m * x + c
    ax.plot(x, y, c='grey', ls='--', label='y={}x+{} ; CCC = {}'.format(round(m,2),round(c,2),round(lccc,4)))
    # Legend
    ax.legend(frameon=False)
    # Save?
    if outdir is not None:
        out_name = os.path.join(outdir,'{}.png'.format(plot_name))
        plt.savefig(out_name)
    # Show
    plt.show()
    return None


def bland_altman_plot(df, ref_col, test_col, dif_percent=True, outdir=None, opt_detail=None, diff_ac=None, *args, **kwargs):
    """DESC:
    Plots paired value sets by Bland-Altman method. Option given between
    raw difference and percent difference for y-axis.
    INPUTS:
    df: Pandas DataFrame, contains data to fit line
    ref_col: str, column name from df, reference set of values
    test_col: str, column name from df, test set of values
    dif_percent: bool, option to use percent difference instead of raw difference
    outdir: str, dir path to save plot in, default None does not save plot
    opt_detail: str, detail to include in plot naming, None does not alter name
    diff_ac: float, absolute value of difference acceptance limits to display
    OUTPUTS:
    displays plot and will save if outdir given
    """
    ## do the math
    ref = np.asarray(df[ref_col])
    test = np.asarray(df[test_col])
    mean = np.mean([ref, test], axis=0)
    if dif_percent:
        diff = (ref-test)/mean
        dif_name = 'Percent Difference'
        rnd = 4
    else:
        diff = ref-test 
        dif_name = 'Difference'
        rnd = 2
    md = np.mean(diff)         # Mean of the difference
    sd = np.std(diff, axis=0)  # Standard deviation of the difference
    CI_low, CI_high = md-1.96*sd, md + 1.96*sd
    ## plot the data
    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md, color='black', linestyle='-')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    if diff_ac is not None:
        plt.axhline(diff_ac, color='red', linestyle='--')
        plt.axhline(-diff_ac, color='red', linestyle='--')
    ## Format and label plot
    name = 'BA Plot {} vs {}'.format(ref_col, test_col)
    if opt_detail is not None:
        name = ' '.join([opt_detail,name])
    plt.title(name)
    plt.xlabel("Means")
    plt.ylabel(dif_name)
    xOutPlot = np.min(mean) + (np.max(mean)-np.min(mean))*1.14
    plt.text(xOutPlot,CI_low,"-1.96SD:\n{}".format(round(CI_low,rnd)),ha = "center",va = "center")
    plt.text(xOutPlot, md, 'Mean:\n{}'.format(round(md,rnd)),ha = "center",va = "center")
    plt.text(xOutPlot,CI_high,"+1.96SD:\n{}".format(round(CI_high,rnd)),ha = "center",va = "center")
    plt.subplots_adjust(right=0.85)
    ## save and display
    if outdir is not None:
        out_name = os.path.join(outdir,'{}.png'.format(name))
        plt.savefig(out_name)
    plt.show()
    return md, sd, mean, CI_low, CI_high