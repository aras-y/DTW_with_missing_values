import numpy as np
import logging

import dtw_missing.missing_utils as mus

logger = logging.getLogger("be.kuleuven.dtw_missing")

def plot_warpingpaths(s1, s2, paths, path=None, filename=None, shownumbers=False, showlegend=False,
                      d=None, figsize=(4, 4),
                      figure=None, matshow_kwargs=None):
    """Plot the warping paths matrix.
    
    Built on DTAIDistance: 
    https://github.com/wannesm/dtaidistance/blob/master/dtaidistance/dtw_visualisation.py

    :param s1: Series 1
    :param s2: Series 2
    :param paths: Warping paths matrix
    :param path: Path to draw (typically this is the best path)
    :param filename: Filename for the image (optional)
    :param shownumbers: Show distances also as numbers
    :param showlegend: Show colormap legend
    :param d: distance to write on the figure
    :param figsize: figure size
    :param figure: Matplotlib Figure object
    :return: Figure, Axes
    """
    try:
        from matplotlib import pyplot as plt
        from matplotlib import gridspec
        from matplotlib.ticker import FuncFormatter
    except ImportError:
        logger.error("The plot_warpingpaths function requires the matplotlib package to be installed.")
        return
    ratio = max(len(s1), len(s2))
    min_y = min(np.nanmin(s1), np.nanmin(s2))
    max_y = max(np.nanmax(s1), np.nanmax(s2))

    if figure is None:
        fig = plt.figure(figsize=figsize, frameon=True)
    else:
        fig = figure
    if showlegend:
        grows = 3
        gcols = 3
        height_ratios = [1, 6, 1]
        width_ratios = [1, 6, 1]
    else:
        grows = 2
        gcols = 2
        height_ratios = [1, 6]
        width_ratios = [1, 6]
    gs = gridspec.GridSpec(grows, gcols, wspace=1, hspace=1,
                           left=0, right=10.0, bottom=0, top=1.0,
                           height_ratios=height_ratios,
                           width_ratios=width_ratios)
    max_s2_x = np.nanmax(s2)
    max_s2_y = len(s2)
    max_s1_x = np.nanmax(s1)
    min_s1_x = np.nanmin(s1)
    max_s1_y = len(s1)

    if path is None:
        # p = dtw.best_path(paths)
        p = None
    elif path == -1 or len(path) == 0:
        p = None
    else:
        p = path

    def format_fn2_x(tick_val, tick_pos):
        return max_s2_x - tick_val

    def format_fn2_y(tick_val, tick_pos):
        return int(max_s2_y - tick_val)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_axis_off()
    
    # if p is not None:
    #     ax0.text(0, 0, "Dist = {:.4f}".format(paths[p[-1][0] + 1, p[-1][1] + 1]))
    if d is not None:
        if isinstance(d, str): # string provided
            ax0.text(0, 0, d)
        else:
            ax0.text(0, 0, "Dist = {:.4f}".format(d))
        
    ax0.xaxis.set_major_locator(plt.NullLocator())
    ax0.yaxis.set_major_locator(plt.NullLocator())

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_ylim([min_y, max_y])
    ax1.set_axis_off()
    ax1.xaxis.tick_top()
    # ax1.set_aspect(0.454)
    ax1.plot(range(len(s2)), s2, ".-")
    ax1.set_xlim([-0.5, len(s2) - 0.5])
    ax1.xaxis.set_major_locator(plt.NullLocator())
    ax1.yaxis.set_major_locator(plt.NullLocator())

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_xlim([-max_y, -min_y])
    ax2.set_axis_off()
    # ax2.set_aspect(0.8)
    # ax2.xaxis.set_major_formatter(FuncFormatter(format_fn2_x))
    # ax2.yaxis.set_major_formatter(FuncFormatter(format_fn2_y))
    ax2.xaxis.set_major_locator(plt.NullLocator())
    ax2.yaxis.set_major_locator(plt.NullLocator())
    ax2.plot(-s1, range(max_s1_y, 0, -1), ".-")
    ax2.set_ylim([0.5, len(s1) + 0.5])

    ax3 = fig.add_subplot(gs[1, 1])
    # ax3.set_aspect(1)
    kwargs = {} if matshow_kwargs is None else matshow_kwargs
    img = ax3.matshow(paths[1:, 1:], **kwargs)
    # ax3.grid(which='major', color='w', linestyle='-', linewidth=0)
    # ax3.set_axis_off()
    if p is not None:
        py, px = zip(*p)
        ax3.plot(px, py, ".-", color="red")
    # ax3.xaxis.set_major_locator(plt.NullLocator())
    # ax3.yaxis.set_major_locator(plt.NullLocator())
    if shownumbers:
        for r in range(1, paths.shape[0]):
            for c in range(1, paths.shape[1]):
                ax3.text(c - 1, r - 1, "{:.2f}".format(paths[r, c]))

    gs.tight_layout(fig, pad=1.0, h_pad=1.0, w_pad=1.0)
    # fig.subplots_adjust(hspace=0, wspace=0)

    if showlegend:
        # ax4 = fig.add_subplot(gs[0:, 2])
        ax4 = fig.add_axes([0.9, 0.25, 0.015, 0.5])
        fig.colorbar(img, cax=ax4) # , label='cost')

    # Align the subplots:
    ax1pos = ax1.get_position().bounds
    ax2pos = ax2.get_position().bounds
    ax3pos = ax3.get_position().bounds
    ax2.set_position((ax2pos[0], ax2pos[1] + ax2pos[3] - ax3pos[3], ax2pos[2], ax3pos[3])) # adjust the time series on the left vertically
    if len(s1) < len(s2):
        ax3.set_position((ax3pos[0], ax2pos[1] + ax2pos[3] - ax3pos[3], ax3pos[2], ax3pos[3])) # move the time series on the left and the distance matrix upwards
        if showlegend:
            ax4pos = ax4.get_position().bounds
            ax4.set_position((ax4pos[0], ax2pos[1] + ax2pos[3] - ax3pos[3], ax4pos[2], ax3pos[3])) # move the legend upwards
    if len(s1) > len(s2):
        ax3.set_position((ax1pos[0], ax3pos[1], ax3pos[2], ax3pos[3])) # move the time series at the top and the distance matrix to the left
        ax1.set_position((ax1pos[0], ax1pos[1], ax3pos[2], ax1pos[3])) # adjust the time series at the top horizontally
        if showlegend:
            ax4pos = ax4.get_position().bounds
            ax4.set_position((ax1pos[0] + ax3pos[2] + (ax1pos[0] - (ax2pos[0] + ax2pos[2])), ax4pos[1], ax4pos[2], ax4pos[3])) # move the legend to the left to equalize the horizontal spaces between the subplots
    if len(s1) == len(s2):
        ax1.set_position((ax3pos[0], ax1pos[1], ax3pos[2], ax1pos[3])) # adjust the time series at the top horizontally
        
    ax = fig.axes

    if filename:
        if type(filename) != str:
            filename = str(filename)
        plt.savefig(filename)
        plt.close()
        fig, ax = None, None
    return fig, ax


def plot_warping(s1, s2, path, filename=None, fig=None, axs=None,
                 series_line_options=None, warping_line_options=None, 
                 warping_line_options_missing=None,
                 missing_time_sample_location='edge', figsize=(8, 4)):
    """Plot the optimal warping between to sequences.

    :param s1: From sequence.
    :param s2: To sequence.
    :param path: Optimal warping path.
    :param filename: Filename path (optional).
    :param fig: Matplotlib Figure object
    :param axs: Array of Matplotlib axes.Axes objects (length == 2)
    :param series_line_options: Dictionary of options to pass to matplotlib plot
        None will not pass any options
    :param warping_line_options: Dictionary of options to pass to matplotlib ConnectionPatch
        None will use {'linewidth': 0.5, 'color': 'orange', 'alpha': 0.8}
    :param warping_line_options_missing: Dictionary of options to pass to matplotlib ConnectionPatch for matches that contain missing values
        None will use {'linewidth': 0.5, 'color': 'orange', 'alpha': 0.8, 'linestyle': (0, (5, 10))}
    :param missing_time_sample_location: Vertical position of the end of a warping line that corresponds to a missing time sample: 
        'edge': nearest edge of the subplot (DEFAULT),
        'middle': centered,
        'interpolation': determined by linear interpolation.
    :param figsize: figure size
    :return: Figure, list[Axes]
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from matplotlib.patches import ConnectionPatch
    except ImportError:
        logger.error("The plot_warp function requires the matplotlib package to be installed.")
        return
    if fig is None and axs is None:
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex='all', sharey='all', figsize=figsize)
    elif fig is None or axs is None:
        raise TypeError(f'The fig and axs arguments need to be both None or both instantiated.')
    
    if missing_time_sample_location == None:
        missing_time_sample_location = 'edge'
    
    if missing_time_sample_location == 'interpolation':
        # from scipy.interpolate import interp1d
        # def interpolate_missing(x): # apply linear interpolation to fill missing values
        #     ind_missing = np.isnan(x)
        #     x_interp = x.copy()
        #     x_interp[np.where(ind_missing)[0]] = interp1d(np.where(~ind_missing)[0], x[~ind_missing])(np.where(ind_missing)[0])
        #     return x_interp
        
        s1_interp = mus.interpolate_missing(s1)
        s2_interp = mus.interpolate_missing(s2)
    
    if series_line_options is None:
        series_line_options = {}
    axs[0].plot(s1, **series_line_options)
    axs[1].plot(s2, **series_line_options)
    plt.tight_layout()
    lines = []
    if warping_line_options is None:
        warping_line_options = {'linewidth': 0.5, 'color': 'orange', 'alpha': 0.8}
    if warping_line_options_missing is None:
        warping_line_options_missing = {'linewidth': 0.5, 'color': 'orange', 'alpha': 0.8, 'linestyle': (0, (5, 10))}
    for r_c, c_c in path:
        if r_c < 0 or c_c < 0:
            continue
        
        yA = s1[r_c]
        yB = s2[c_c]
        if np.isnan(yA) or np.isnan(yB):
            lno = warping_line_options_missing
        else:
            lno = warping_line_options
        if np.isnan(yA):
            if missing_time_sample_location == 'edge':
                yA = axs[0].get_ylim()[0] # use the bottom of the vertical axis when the sample is missing
            elif missing_time_sample_location == 'middle':
                yA = np.mean(axs[0].get_ylim()) # use the center of the vertical axis when the sample is missing
            elif missing_time_sample_location == 'interpolation':
                yA = s1_interp[r_c] # determine the vertical position by linear interpolation
            
        if np.isnan(yB):
            if missing_time_sample_location == 'edge':
                yB = axs[1].get_ylim()[1] # use the top of the vertical axis when the sample is missing
            elif missing_time_sample_location == 'middle':
                yB = np.mean(axs[1].get_ylim()) # use the center of the vertical axis when the sample is missing
            elif missing_time_sample_location == 'interpolation':
                yB = s2_interp[c_c] # determine the vertical position by linear interpolation
        
        con = ConnectionPatch(xyA=[r_c, yA], coordsA=axs[0].transData,
                              xyB=[c_c, yB], coordsB=axs[1].transData, **lno)
        lines.append(con)
    for line in lines:
        fig.add_artist(line)
    if filename:
        plt.savefig(filename)
        plt.close()
        fig, axs = None, None
    return fig, axs