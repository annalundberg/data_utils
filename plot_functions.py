# -*- coding: utf-8 -*-
"""
Script for generating various types of distribution plots. Plot types
include: boxplot, boxplot w/ jitter, swarmplot, density curve. Includes a
stat labeling function to show mean & st deviation for boxplot or swarmplot.

Created Nov 2020

@author: anna.lundberg
"""

##############################################################################
######################## Import Libraries ####################################
##############################################################################

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


##############################################################################
######################## Function Block ######################################
##############################################################################

def get_stat_labels(data_frame, depend_var, group):
    '''DESC
    Generate mean and standard deviation stats for plot labeling
    INPUTS
    data_frame: pandas dataframe (df) containing data for plot_labels
    depend_var: str(df.column_name) variable to generate stats for
    group:  str(df.column_name) categorical variable for grouping
    '''
    means = data_frame.groupby([group], sort=False)[depend_var].mean().values
    sds = data_frame.groupby([group], sort=False)[depend_var].std(ddof=0).values
    if means.max() < 100:
        mean_labels = [str(np.round(m, 2)) for m in means]
        sd_labels = [('+/-'+str(np.round(s, 2))) for s in sds]
    else:
        mean_labels = [str(int(np.round(m, 0))) for m in means]
        sd_labels = [('+/-'+str(int(np.round(s, 0)))) for s in sds]
    return mean_labels, sd_labels


def make_boxplot(data_frame, depend_var, group, sub_plot, color_map='winter',
                 subgroup=None, stat_label=False, jitter=False,
                 scale=False):
    '''DESCS:
    Generates a boxplot using pandas dataframe as a source and plots with
    seaborn over matplotlib's pyplot. Optional Mean with St Dev labeling
    available when number of x-axis categories is at most 16. Log base 10
    scaling friendly.
    INPUTS:
    data_frame: pandas dataframe (df) containing data to be plotted
    depend_var: str(df.column_name) variable to be plotted on y-axis
    group:  str(df.column_name) categorical variable to set x-axis grouping
    sub_plot:   pyplot axes object, frame for figure to be plotted in
    color_map:  matplotlib colormap, default(winter) is Epic colors friendly
    subgroup:   str(df.column_name) optional categorical variable to set hue by
    lim:    bool, set y-axis minimum to 0 (True/False)
    stat_label: bool, label by x axis with mean +/- stdev if num(x) <= 16
    jitter:  bool, plot jittered datapoints with boxplot
    scale: bool, transform y axis to log10 scale
    OUTPUTS:
    box_plot:   seaborn generated boxplot, pyplot figure
    '''
    data_frame[depend_var].dropna(inplace=True)
    sns.set_style('whitegrid')
    if scale:
        sub_plot.set(yscale='log')
    box_plot = sns.boxplot(x=group, y=depend_var, hue=subgroup,
                               data=data_frame, palette=color_map,
                               ax=sub_plot, width=0.75)
    if subgroup is None and stat_label and len(pd.unique(data_frame[group])) <= 16:
        mean_labels, sd_labels = get_stat_labels(data_frame, depend_var, group)
        for tick, label in enumerate(mean_labels):
            sub_plot.text(tick, data_frame[depend_var].max(), label,
                            horizontalalignment='center', size='small',
                            color='k', weight='semibold')
        for tick, label in enumerate(sd_labels):
            sub_plot.text(tick, 
                            (data_frame[depend_var].max()-0.5*data_frame[depend_var].std()), 
                            label, horizontalalignment='center', 
                            size='small', color='k', weight='semibold')
    if jitter and subgroup is None:
        box_plot = sns.stripplot(y=depend_var, x=group, ax=sub_plot, 
                                 data=data_frame, jitter=0.3, alpha=0.7, palette='magma', size=3)
    elif jitter:
        box_plot = sns.stripplot(y=depend_var, x=group, hue=subgroup, ax=sub_plot, data=data_frame, 
                                 jitter=0.3, alpha=0.7, palette='magma', size=3, dodge=True)
        handles, labels = box_plot.get_legend_handles_labels()
        plt.legend(handles[0:len(pd.unique(data_frame[subgroup]))],labels[0:len(pd.unique(data_frame[subgroup]))])
    ## rotate x labels for larger dataset
    if len(pd.unique(data_frame[group])) > 5:
        box_plot.set_xticklabels(box_plot.get_xticklabels(), rotation=90)
    return box_plot


def make_swarmplot(data_frame, depend_var, group, sub_plot, color_map='winter',
                   subgroup=None, stat_label=False, scale=False):
    '''DESCS:
    Generates a beeswarm plot using pandas dataframe as a source and plots 
    with seaborn over matplotlib's pyplot. Optional Mean with St Dev labeling
    available when number of x-axis categories is at most 16. Log base 10
    scaling friendly.
    INPUTS:
    data_frame: Pandas DataFrame, containing data to be plotted
    depend_var: str(df.column_name), variable to be plotted on y-axis
    group: str(df.column_name), categorical variable to set x-axis grouping
    sub_plot: pyplot axes object, frame for figure to be plotted in
    color_map: matplotlib colormap, default(winter) is Epic colors friendly
    subgroup: str(df.column_name) optional categorical variable to set hue by
    stat_label: bool, label by x axis with mean +/- stdev if num(x) <= 16
    scale: bool, transform y axis to log10 scale
    OUTPUTS:
    plot: pyplot figure, seaborn generated swarmplot
    '''
    if subgroup is not None:
        stat_label = False
    data_frame.dropna(inplace=True)
    sns.set_style('whitegrid')
    plot = sns.swarmplot(x=group, y=depend_var, hue=subgroup, data=data_frame,
                         palette=color_map, ax=sub_plot, size=2)
    plot.set_xticklabels(plot.get_xticklabels(), rotation=90)
    if stat_label and len(pd.unique(data_frame[group])) <= 16:
        ## get mean & st.dev labels on bps ##
        mean_labels, sd_labels = get_stat_labels(data_frame, depend_var, group)
        for tick, label in enumerate(mean_labels):
            sub_plot.text(tick, data_frame[depend_var].max(), label,
                          horizontalalignment='center', size='small',
                          color='k', weight='semibold')
        for tick, label in enumerate(sd_labels):
            sub_plot.text(tick, 
                          (data_frame[depend_var].max()-0.5*data_frame[depend_var].std()), 
                          label, horizontalalignment='center', 
                          size='small', color='k', weight='semibold')
    if scale:
        plot.set(yscale='log')
    return plot


def make_density_curve(df, depend_var, group, sub_plot, groups=None, title=None, 
                        color_map='winter', scale=False):
    '''DESC
    Makes a single plot with 1+ density curves. Input dataframe is subset by input 
    group to make a density curve showing the distribution of the input metric. Log 
    base 10 scaling friendly.
    INPUTS:
    df: pandas DataFrame, containing data to be plotted
    depend_var: str(df.column_name), variable to be plotted
    group: str(df.column_name), grouping variable for density curves
    sub_plot: pyplot axes object, frame for figure to be plotted in
    groups: list of str, subset of pd.unique(df[group]) to be included in plot
    title: str, custom title to add to plot
    color_map: matplotlib colormap, default(winter) is Epic colors friendly
    scale: bool, transform x-axis to log10 scale
    OUTPUTS:
    dc: pyplot figure, seaborn generated density curves
    '''
    if groups is None:
        groups = pd.unique(df[group])
    colors = sns.color_palette(color_map, len(groups))
    for i in range(len(groups)):
        group_df = df[df[group] == groups[i]]
        dc = sns.distplot(group_df[depend_var], hist=False, kde=True, ax=sub_plot,
                            color=colors[i], kde_kws={'linewidth':2}, label=groups[i])
    handles,labels = sub_plot.get_legend_handles_labels()
    dc = plt.legend(handles,labels,prop={'size':10})
    if title is not None:
        dc = plt.title(title)
    if scale:
        dc.set(xscale='log')
    return dc
