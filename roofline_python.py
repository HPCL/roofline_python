
import io
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import argparse
import numpy as np


def SetupArgs():
    """
    Setup the arguments required to run the script on the command line
    :return:
    """
    parser = argparse.ArgumentParser()
    # input roofline.json file
    parser.add_argument("--input", "-i", help="Path to the roofline JSON file", required=True)
    # config file for the graph
    parser.add_argument("--graphconfig", "-g", help="The config file path/name for specifying graph requirements", required=True)
    args = parser.parse_args()

    return args

#
def args_to_vars(args):
    """
    This takes the command line arguments and turns the into variables to pass to the library functions
    This makes it so that those same functions can be called from a jupyter notebook with just the correct variables
    :param args:
    :return:
    """
    input_filename = args.input
    graph_filename = args.graphconfig
    return input_filename, graph_filename


def read_graph_configs(graph_filename):
    """
    Load the graph config file (json) into a dictionary
    :param graph_filename:
    :return:
    """
    # Load the json file
    with open(graph_filename) as filename:
        graph_config = json.load(filename)

    return graph_config


# create the base roofline chart
def create_roofline(input_filename, graph_filename):
    """
    Parse the json into the data we want to plot on the rooflines
    :param input_filename:
    :param graph_filename:
    :return:
    """
    # Load the json file
    with open(input_filename) as filename:
        data = json.load(filename)

    # Parse the gbytes section
    gbytes = data['empirical']['gbytes']
    # Load gbytes data into Pandas dataframe
    gbytes_df = pd.DataFrame(gbytes['data'], columns=['name', 'slope'])
    gbytes_df['slope'] = gbytes_df['slope']
    # Add the default x and y intercept
    gbytes_df['x'] = 0
    gbytes_df['y'] = 0

    # Parse the gflop data section
    gflops = data['empirical']['gflops']
    # Load gflops data into a Pandas dataframe
    gflops_df = pd.DataFrame(gflops['data'], columns=['name', 'y'])
    # Add the default x intercept and "slope" (really the value)
    gflops_df['x'] = 100
    gflops_df['slope'] = gflops_df['y']

    # Calculate the point where each memory line meets the peak line, add to dataframe
    rows = len(gbytes_df)
    peak = max([p for p in gflops_df['y']])
    new_xes = []
    for row in range(rows):
        name = gbytes_df['name'][row]
        x = peak / gbytes_df['slope'][row]
        gbytes_df.loc[len(gbytes_df.index)] = [name, gbytes_df['slope'][row], x, peak]
        new_xes.append(x)

    # Find the x coordinates to start the roof lines at
    peak_mem = max([m for m in gbytes_df['slope']])
    peak_rows = len(gflops_df)
    for i in range(peak_rows):
        x_val = gflops_df['y'][i] / peak_mem
        gflops_df.loc[len(gflops_df)] = [gflops_df['name'][i], gflops_df['y'][i], x_val, gflops_df['y'][i]]

    # Make the label columns for the graph
    gbytes_df['label'] = gbytes_df['name'] + ' ' + gbytes_df['slope'].astype(str) + ' ' + 'GB/s'
    gflops_df['label'] = gflops_df['name'] + ' ' + gflops_df['slope'].astype(str) + ' ' + 'GFLOPs/s'

    # Make type column for identifying later
    gbytes_df['type'] = 'memory' 
    gflops_df['type'] = 'peak'

    # Concatenate the gbyte and gflop data into one dataframe to plot
    g_df = pd.concat([gflops_df, gbytes_df], ignore_index=True)

    graph_config = read_graph_configs(graph_filename)

    # *** BEGIN PLOTTING *** #
    if (graph_config['table']):
        # get table of configs
        metadata_df = add_table(gflops)
        plot_table(g_df, gbytes_df, gflops_df, metadata_df, graph_config)
    else:
        plot_no_table(g_df, gbytes_df, gflops_df, graph_config)


def calculate_axes(g_df):
    """
    Determine the min/max values for both the x and y axes
    :param g_df:
    :return:
    """
    xmin = 0.1
    xmax = 100.00
    ymin = 1 ** int(math.floor(math.log10(g_df['slope'][0]*xmin)))
    ymax = (ymin*10) ** int(math.floor(math.log10(g_df['slope'][0]*10)))
    ymin = 0.0000001
    alpha = 1.15

    return xmin, xmax, ymin, ymax, alpha


def plot_flop_labels(ax, gflops_df, xmax, alpha):
    """
    Plot the labels for the peak flop lines
    :param ax:
    :param gflops_df:
    :param xmax:
    :param alpha:
    :return:
    """
    for i in range(len(gflops_df.name.unique())):
        ax.text(xmax, gflops_df['y'][i]*alpha, gflops_df['label'][i], size='medium', ha="right")


def plot_mem_labels(ax, gbytes_df, xmin, alpha):
    """
    Plot the labels for the peak memory lines
    :param ax:
    :param gbytes_df:
    :param xmin:
    :param alpha:
    :return:
    """
    for j in range(len(gbytes_df.name.unique())):
        mem = gbytes_df['name'][j]
        (xmax, slope) = max([(gbytes_df['x'][i],gbytes_df['slope'][i]) for i in range(len(gbytes_df['x'])) if gbytes_df['name'][i]==mem])
        ylab = slope * xmin
        ytop = slope * xmax
        #transform the data to coordinates for plotting the correct angle
        pa = ax.transData.transform_point((xmin, ylab))
        p = ax.transData.transform_point((xmax, ytop))
        ang = np.arctan2(p[1]-pa[1], p[0]-pa[0])
        trans_angle = np.rad2deg(ang)
        ax.text(xmin, ylab*alpha, gbytes_df['label'][j], size='medium', rotation=trans_angle)


def plot_table(g_df, gbytes_df, gflops_df, metadata_df, graph_config):
    """
    Create a roofline plot with a table of metadata
    :param g_df:
    :param gbytes_df:
    :param gflops_df:
    :param graph_config:
    :return:
    """

    xmin, xmax, ymin, ymax, alpha = calculate_axes(g_df)

    # set some general plot settings
    sns.set(rc={'figure.figsize':(12,8)})
    palette = sns.color_palette( "Dark2", int(len(g_df)/2))
    sns.set(font_scale=1.45)

    # plot the lines and peak flop labels
    fig, (ax,ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]})
    sns.lineplot(data=g_df, x="x", y="y", hue="label",palette=palette, ax=ax, linewidth=3, legend = False)

    # set axes labels and values, set log scale
    ax.set(xlabel=graph_config["xlabel"], ylabel=graph_config["ylabel"])
    ax.set(xscale="log", yscale="log", xlim=(xmin, xmax), ylim=(ymin,ymax))

    # plot the line label(s)
    plot_flop_labels(ax, gflops_df, xmax, alpha)
    plot_mem_labels(ax, gbytes_df, xmin, alpha)

    if graph_config["appdata"]:
        # plot the application information
        add_application_data(ax, graph_config["appdata"])

    # plot table
    bbox = [0,0,1,1]
    ax2.axis('off')
    ax2.axis('tight')
    mpl_table = ax2.table(cellText=metadata_df.values, bbox=bbox, colLabels=metadata_df.columns, edges='horizontal')
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(10)

    # add grid lines, title, legend
    ax.grid(b=True, which='both')
    if graph_config["title"]:
        title = "Empirical Roofline Graph"
        ax.set_title(title, fontsize=20)

    # finalize and save the file
    fig.tight_layout()
    ax.figure.savefig(graph_config["outfile"])
    print("Figure Saved as", graph_config["outfile"])

def plot_no_table(g_df, gbytes_df, gflops_df, graph_config):
    """
    Create a roofline plot without a table of metadata
    :param g_df:
    :param gbytes_df:
    :param gflops_df:
    :param graph_config:
    :return:
    """

    xmin, xmax, ymin, ymax, alpha = calculate_axes(g_df)

    # set some general plot settings
    sns.set(rc={'figure.figsize':(12,8)})
    sns.set(font_scale=graph_config["font_scale"])
    palette = sns.color_palette("Dark2", int(len(g_df)/2))

    # plot the lines and peak flop labels
    ax = sns.lineplot(data=g_df, x="x", y="y", hue="label", palette=palette, linewidth=3, legend = False)
    ax.set(xlabel=graph_config["xlabel"], ylabel=graph_config["ylabel"])
    ax.set(xscale="log", yscale="log", xlim=(xmin, xmax), ylim=(ymin,ymax))
    # add grid lines
    ax.grid(b=True, which='both')

    # plot the line label(s)
    plot_flop_labels(ax, gflops_df, xmax, alpha)
    plot_mem_labels(ax, gbytes_df, xmin, alpha)

    if graph_config["appdata"]:
        # plot the application information
        add_application_data(ax, graph_config["appdata"])

    if graph_config["title"]:
        ax.set_title(graph_config["title"], fontsize=20)

    # finalize and save the file
    ax.figure.tight_layout()
    ax.figure.savefig(graph_config["outfile"])
    print("Figure Saved as", graph_config["outfile"])


def add_table(gflops):
    """
    Add the table next to the chart from config metadata
    :param gflops: the dictionary containing peak flop data (parsed from json)
    :return: a dataframe containing the desired metadata
    """
    # Get the host name
    metad = gflops['metadata']['']
    host_info = metad[5]
    host_list = host_info.split("'")
    host_name = host_list[1]

    # parse the config section to add to the table
    for i in range(len(metad)):
        conf = metad[i]
        if conf.startswith('CONFIG'):
            config = metad[i]
            break
    config_list = config.split("'")
    config_list = config_list[1:]

    config_for_pd = [('HOST',host_name)]

    config_name = ''
    value_list = []
    for x in range(len(config_list)):
        cur_conf = config_list[x]
        # hit a key
        if cur_conf.isupper():
            config_name = cur_conf
        # hit a value
        elif re.search('[a-zA-Z0-9]', cur_conf):
            value_list.append(cur_conf)
        # end of a section add config and value to dict
        elif ("]," in cur_conf)or ("]}" in cur_conf):
            if len(value_list)>0:
                config_for_pd.append((config_name, "".join(value_list)))
            value_list = [] # reset

    metadata_df = pd.DataFrame(config_for_pd, columns=["Config", "Value"])
    return metadata_df


def add_application_data(ax, app_data_filename):
    """
    Plot the application performance data on the roofline chart
    :param app_data_filename: the file name containing the application data
    """
    # Load the csv file
    with open(app_data_filename) as csv_filename:
        app_df = pd.read_csv(csv_filename)

    app_df['Gflops/Sec'] = (app_df['Total Flops']/app_df['Time (s)'])/1000000000
    #app_df['Gflops/Sec'] = app_df['Total Flops']/app_df['Time (s)']
    print(app_df)
    sns.scatterplot(ax=ax, x=app_df['Arithmetic Intensity'], y=app_df['Gflops/Sec'], style=app_df['Label'], hue=app_df['Label'], s=150)
    #ax.legend(bbox_to_anchor=(1.125, 1), loc='upper left')
    ax.legend(loc='lower right')


if __name__ == '__main__':
    args = SetupArgs()
    (input_filename, graph_filename) = args_to_vars(args)
    print("Generating roofline graph for:", input_filename, "\nWith config file: ", graph_filename)
    create_roofline(input_filename, graph_filename)
