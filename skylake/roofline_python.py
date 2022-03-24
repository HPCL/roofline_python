
import io
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
# import plotly.graph_objects as go
import argparse
# matplotlib.use('svg') # TODO: add other output types?


def SetupArgs():
    parser = argparse.ArgumentParser()
    # input roofline.json file
    parser.add_argument("--input", "-i", help="Path to the roofline JSON file", required=True)
    # output file name for saving the roofline chart
    parser.add_argument("--outfile", "-o", help="The output file path/name", required=True)
    # optional application data file
    parser.add_argument("--appdata", "-a", help="Path to the CSV file that contains application performance data")
    parser.add_argument("--table", "-t", help="Include the config information in a table next to the roofline plot", action="store_true")
    args = parser.parse_args()

    return args

# create the base roofline chart
def create_roofline(args):
    # Load the json file
    with open(args.input) as filename:
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

    # *** BEGIN PLOTTING *** #
    if (args.table):
        plot_table(g_df, gbytes_df, gflops_df)
    else:
        plot_no_table(g_df, gbytes_df, gflops_df)

def plot_table(g_df, gbytes_df, gflops_df):
    # calculate the axes scale
    xmin = 0.01
    xmax = 100.00
    ymin = 10 ** int(math.floor(math.log10(g_df['slope'][0]*xmin)))
    ymax = ymin ** int(math.floor(math.log10(g_df['slope'][0]*10)))

    #calculate the midpoints for labels
    xmid = math.sqrt(xmin * xmax)
    ymid = g_df['slope'][0] * xmid
    y0gbytes = ymid
    x0gbytes = y0gbytes/g_df['slope'][0]
    alpha = 1.065

    # set some general plot settings
    sns.set(rc={'figure.figsize':(12,8)})
    palette = sns.color_palette( "Dark2", int(len(g_df)/2))

    font_size=10

    # plot the lines and peak flop labels
    fig, (ax,ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]})
    sns.lineplot(data=g_df, x="x", y="y", hue="label",palette=palette, ax=ax, linewidth=3)

    # set axes labels and values, set log scale
    ax.set(xlabel='FLOPs / Byte', ylabel='GFLOPs / Second')
    ax.set(xscale="log", yscale="log", xlim=(xmin, xmax), ylim=(ymin,ymax))

    # plot the line label(s)
    for i in range(len(gflops_df.name.unique())):
        ax.text(xmax, y0gbytes*alpha, g_df['label'][i], size='medium', ha="right")
    for j in range(len(gbytes_df.name.unique())):
        mem = gbytes_df['name'][j]
        (xmax, slope) = max([(gbytes_df['x'][i],gbytes_df['slope'][i]) for i in range(len(gbytes_df['x'])) if gbytes_df['name'][i]==mem])
        xmid = math.sqrt(xmin * xmax)
        ymid = slope * xmid
        y0gbytes = ymid
        x0gbytes = y0gbytes/slope
        alpha = 1.25
        angle = math.degrees(math.atan(slope))/2
        ax.text(x0gbytes, y0gbytes*alpha, gbytes_df['label'][j], size='medium', rotation=52)

    if args.appdata:
        app_df = add_application_data(args)
        # plot the application information
        sns.scatterplot(x=app_df['Arithmetic Intensity'], y=app_df['Gflops/Sec'], ax=ax, style=app_df['Label'], hue=app_df['Label'], s=100)

    # add table of configs
    metadata_df = add_table(gflops)
    # plot table
    bbox = [0,0,1,1]
    ax2.axis('off')
    ax2.axis('tight')
    mpl_table = ax2.table(cellText=metadata_df.values, bbox=bbox, colLabels=metadata_df.columns, edges='horizontal')
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    # add grid lines, title, legend
    ax.grid(b=True, which='both')
    title = "Empirical Roofline Graph"
    ax.set_title(title, fontsize=20)
    ax.legend(loc='lower right')

    fig.tight_layout()

    # save the file
    ax.figure.savefig(args.outfile)
    print("Figure Saved as", args.outfile)

def plot_no_table(g_df, gbytes_df, gflops_df):
    # calculate the axes scale
    xmin = 0.01
    xmax = 100.00
    # ymin = 10 ** int(math.floor(math.log10(g_df['slope'][0]*xmin)))
    ymin = 1.0
    # ymax = ymin ** int(math.floor(math.log10(g_df['slope'][0]*10)))
    ymax = 10 ** int(math.floor(math.log10(g_df['slope'][0]*10)))

    #calculate the midpoints for labels
    xmid = math.sqrt(xmin * xmax)
    ymid = g_df['slope'][0] * xmid
    y0gbytes = ymid
    x0gbytes = y0gbytes/g_df['slope'][0]
    alpha = 1.065

    # set some general plot settings
    sns.set(rc={'figure.figsize':(12,8)})
    palette = sns.color_palette( "Dark2", int(len(g_df)/2))

    # plot the lines and peak flop label
    ax = sns.lineplot(data=g_df, x="x", y="y", hue="label", palette=palette, linewidth=3)
    ax.set(xlabel='FLOPs / Byte', ylabel='GFLOPs / Second')
    ax.set(xscale="log", yscale="log", xlim=(xmin, xmax), ylim=(ymin,ymax))

    # plot the line label(s)
    for i in range(len(gflops_df.name.unique())):
        ax.text(xmax, gflops_df['y'][i]*alpha, g_df['label'][i], size='medium', ha="right")
    for j in range(len(gbytes_df.name.unique())):
        mem = gbytes_df['name'][j]
        (xmax, slope) = max([(gbytes_df['x'][i],gbytes_df['slope'][i]) for i in range(len(gbytes_df['x'])) if gbytes_df['name'][i]==mem])
        xmid = math.sqrt(xmin * xmax)
        ymid = slope * xmid
        y0gbytes = ymid
        x0gbytes = y0gbytes/slope
        alpha = 1.1
        angle = (math.degrees(math.atan(slope))/2)-13
        ax.text(x0gbytes, y0gbytes*alpha, gbytes_df['label'][j], size='medium', rotation=angle)

    if args.appdata:
        app_df = add_application_data(args)
        # plot the application information
        ax = sns.scatterplot(x=app_df['Arithmetic Intensity'], y=app_df['Gflops/Sec'], style=app_df['Label'], hue=app_df['Label'], s=100)

    # add grid lines, title, legend
    ax.grid(b=True, which='both')
    title = "Empirical Roofline Graph"
    ax.set_title(title, fontsize=20)
    ax.legend(loc='lower right')

    ax.get_figure().tight_layout()

    # save the file
    ax.figure.savefig(args.outfile)
    print("Figure Saved as", args.outfile)


# add the table next to the chart from config metadata
def add_table(gflops):
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
        elif ("]," in cur_conf) or ("]}" in cur_conf):
            if len(value_list)>0:
                config_for_pd.append((config_name, "".join(value_list)))
            value_list = [] # reset

    metadata_df = pd.DataFrame(config_for_pd, columns=["Config", "Value"])
    return metadata_df


# plot the application performance data on the roofline chart
def add_application_data(args):
    # Load the csv file
    with open(args.appdata) as csv_filename:
        app_df = pd.read_csv(csv_filename)

    app_df['Gflops/Sec'] = (app_df['Total Flops']/app_df['Time (s)'])/1000000000

    return app_df


if __name__ == '__main__':
    args = SetupArgs()
    print(args)
    create_roofline(args)