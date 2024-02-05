import re, sys
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn import linear_model


def scatter(x, y, png_fn=None, ax=None, xlab=None, ylab=None, title=None, text=None, xyline=False,
        linregress=True, linregress_label=True, coeff_determination=False, s=None, color=None, heatscatter=False):
    '''
    Make a simple scatter plot. 
    Able to add text. 

    Parameters
    ----------
    x : array_like
        x axis data
    y : array_like
        y axis data
    png_fn : str, optional: png_fn or ax 
        output png file name
    ax : axis object, optional: png_fn or ax 
    xlab : str
    ylab : str
    title : str
    text : str
        add text to figure
    xyline: bool
        add y = x dashed line
    linregress : boolen, default=True
        whether to create a linear regression line with Pearson correlation
    coeff_determination : boolen, dafault=True
        whether to compute coefficient of determination (R^2) 
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
        https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression
    s : float, marker size
    color : str, marker color
    heatscatter: boolen
        add KDE 

    Returns
    -------

    Notes
    -----

    '''
    if png_fn:
        fig, ax = plt.subplots()
    elif ax:
        pass
    else:
        sys.exit('Either png_fn or ax must be provided!\n')
    if not color:
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]

    x = np.array(x)
    y = np.array(y)
    if not pd.api.types.is_integer_dtype(x):
        x[(x == np.inf) | (x == -np.inf)] = np.nan
    if not pd.api.types.is_integer_dtype(y):
        y[(y == np.inf) | (y == -np.inf)] = np.nan
    removed = ( np.isnan(x) | np.isnan(y) )
    x = x[~removed]
    y = y[~removed]

    if heatscatter:
        # adopt from https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib
        # calculae the point density
        try:
            xy = np.vstack([x,y])
            z = stats.gaussian_kde(xy)(xy)
            # sort the points by density, so that the densest points are plotted last
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]
            sc = ax.scatter(x, y, s=s, c=z, cmap='cividis')
        except:
            ax.scatter(x, y, s=s, color=color)
        if removed.sum() > 0:
            ax.text(0.2, 0.92, f'{removed.sum()} Inf/NA points removed', transform=ax.transAxes)
        #cbar = plt.colorbar(sc, ax=ax)
        #cbar.ax.set_ylabel('Density', rotation=270)
    else:
        ax.scatter(x, y, s=s, color=color)
        if removed.sum() > 0:
            ax.text(0.2, 0.92, f'{removed.sum()} Inf/NA points removed', transform=ax.transAxes)

    if xyline:
        ax.plot(x, x, color='0.7', linestyle='--', alpha=0.6)  # Plot y = x line with dashed linestyle

    if xlab:
        ax.set_xlabel(xlab)
    if ylab:
        ax.set_ylabel(ylab)
    if title:
        ax.set_title(title)
    if text:
        ax.text(0.2, 1.02, text, transform=ax.transAxes)
    if linregress:
        slope, intercept, r, p, stderr = stats.linregress(x, y)
        r, p = stats.pearsonr(x, y)
        if slope > 0:
            line = f'y={intercept:.2g}+{slope:.2g}x, r={r:.2g}, p={p:.2g}'
        else:
            line = f'y={intercept:.2g}{slope:.2g}x, r={r:.2g}, p={p:.2g}'
        if coeff_determination:
            # Create linear regression object
            regr = linear_model.LinearRegression()
            # Train the model
            regr.fit(np.array(x).reshape((-1, 1)), np.array(y).reshape((-1, 1)))
            # compute R2
            r2 = regr.score(np.array(x).reshape((-1, 1)), np.array(y).reshape((-1, 1)))
            line = line + f', R^2={r2:.2g}'
        if linregress_label:
            ax.plot(x, intercept + slope * x, '0.8', label=line, zorder=10)
            ax.legend(fontsize='small')
        else:
            ax.plot(x, intercept + slope * x, '0.8', zorder=10)

    if png_fn:
        fig.savefig(png_fn)


def format_e(x):
    '''
    Format scientific number to mathematic expression for plot.
    e.g. 2.00e+05 to 2.00x10^05 in mathematic expression

    Patameters
    ----------
    x : str
        a scientific number

    Returns
    -------
    y : str
        mathematic expression for text in plot

    Notes
    -----
    x == 0: return float part of x
    power == -1|-2: return float format of x
    '''

    # check x is a number
    try:
        float(x)
    except:
        sys.stderr.write('ERROR: %s is not a number!\n'%(x))
        sys.exit()

    x_ls = re.split('[Ee]', x)
    x_ls[1] = re.sub('\+', '', x_ls[1])

    # when x == 0
    if float(x) == 0:
        y = x_ls[0]
        return y

    if -3 < int(x_ls[1]) < 0:
        # if the power if -1 or -2, change to float
        y = str(float(x))
    else:
        y = '$'+x_ls[0]+r'\times10^{'+x_ls[1]+'}'+'$'

    return y


def snsbox_get_x(x_categories, hue_categories, width=0.8):
    '''
    Get x coordinates of boxes in Seaborn boxplot.
    [category1:hue1, category1:hue2, category2:hue1, category2:hue2]
    x_categories : int
        number of categories on x axis
    hue_categories : int
        number of hue categories
    width : float (default 0.8)
        width of all the elements for one level of the major grouping variable. (from seaborn.boxplot with default .8)
    '''
    #print(x_categories, hue_categories, width)
    element_length = width / hue_categories
    if hue_categories == 1:
        return np.arange(x_categories)
    elif hue_categories % 2 == 0:
        shifts = np.arange(hue_categories / 2) * element_length + element_length / 2
        shifts = list(np.flip(shifts*(-1))) + list(shifts)
    else:
        shifts = (np.arange((hue_categories-1) / 2)+1) * element_length
        shifts = list(np.flip(shifts*(-1))) + [0] + list(shifts)
    shifts = [[x] for x in shifts]
    xs = np.repeat(np.atleast_2d(np.arange(x_categories)), hue_categories, axis=0)+np.array(shifts)
    xs = xs.T.flatten()
    return xs
