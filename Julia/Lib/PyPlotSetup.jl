module PyPlotSetup

export setuppyplot2d!, setuppyplot3d!

# ==============================================================================

function setuppyplotcommon!(plt::Module)::Nothing
    plt.rc("figure", dpi=600)
    plt.rc("legend", framealpha=1.0)  # legend frame and background opacity
    plt.rc("axes", axisbelow=true)    # grid lines in background behind plots
    plt.rc("grid", linewidth=0.5)     # width of background grid lines
    plt.rc("patch", linewidth=0.75)   # width of legend border, see: https://stackoverflow.com/questions/38359008/matplotlib-is-there-an-rcparams-way-to-adjust-legend-border-width
    # Alternatively:
    # rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    # rcParams["figure.dpi"] = 600
    # rcParams["font.size"] = SMALL_SIZE
    # etc.
    return
end

function setuppyplot2d!(plt::Module)::Nothing
    setuppyplotcommon!(plt)
    SMALL_SIZE = 7
    MEDIUM_SIZE = 8
    BIGGER_SIZE = 9
    plt.rc("font", size=SMALL_SIZE)          # controls default text sizes
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rc("axes", titlesize=MEDIUM_SIZE)    # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)    # legend fontsize
    return
end

function setuppyplot3d!(plt::Module)::Nothing
    setuppyplotcommon!(plt)
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16
    plt.rc("font", size=SMALL_SIZE)          # controls default text sizes
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rc("axes", titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)    # legend fontsize
    return
end

# ==============================================================================
end
