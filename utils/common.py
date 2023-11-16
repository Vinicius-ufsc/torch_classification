import os

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
        
import math
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import numpy as np
import cv2 as cv
import textwrap
import os

"""
Plot function.
"""

def plot_image_grid(image_paths, ncols = 1, kmax=None, image_titles=None, 
                    grid_title=None, high_res = False, wrap=40, 
                    hspace = -0.65, wspace = -0.30, resize = False, 
                    resize_size = (160,160)):
    
    """
    high_res : change figsize from 3,6 to 7,14 -> more resolution.
    """
    
    if type(image_paths) is list:
        if ncols == 1:
            ncols = 4 if len(image_paths) > 4 else len(image_paths)
    else:
        image_paths = [image_paths]
    
    if kmax is not None:
        image_paths = image_paths[:kmax]

    # calculate nrows given length of image_paths and ncols
    nrows = math.ceil(len(image_paths) / ncols)

    # adjust figsize given nrows and ncols
    figsize = (ncols * 3, nrows * 6) # 3,6
    
    if high_res:
        figsize = (ncols * 7, nrows * 14) # 3,6

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor='lightgray')
    fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = hspace, wspace = wspace)
    
    # hide axis and remove blank spaces
    if ncols > 1:
        for ax in axes.flatten():
            ax.set_axis_off()
            ax.margins(0)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
    else:
        axes.set_axis_off()
        axes.margins(0)
        axes.xaxis.set_major_locator(plt.NullLocator())
        axes.yaxis.set_major_locator(plt.NullLocator())

    for i, image_path in enumerate(image_paths):
        if type(image_paths[0]) is str:
            img = plt.imread(image_path)
        elif type(image_paths[0]) is Image.Image:
            img = np.array(image_path)
        else:
            img = image_path
            
        if resize is True:
            img = cv.resize(img, (resize_size))

        if ncols > 1:
            axes.ravel()[i].imshow(img)
            axes.ravel()[i].set_axis_off()
        else:
            axes.imshow(img)
            axes.set_axis_off()

        if image_titles is not None:
            
            title = textwrap.wrap(image_titles[i], wrap)
            title = "\n".join(title)
            
            if ncols > 1:
                axes.ravel()[i].set_title(title, fontsize=10)
            else:
                axes.set_title(title, fontsize=10)

        # add label as title below each image
        if type(image_paths[0]) is str and image_titles is None:
            if ncols > 1:
                axes.ravel()[i].set_title(str(Path(image_path).stem), fontsize=10)
            else:
                axes.set_title(str(Path(image_path).stem), fontsize=10)

    if grid_title is not None:
        fig.suptitle(grid_title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()
    
    
import numpy as np
import matplotlib.pyplot as plt

def scatter_plot_with_labels_and_centers(data, labels, figsize, title='scatter', class_dict = None):
    # data: a numpy array of shape (N, 2) containing the x and y coordinates of the points
    # labels: a numpy array of shape (N,) containing the cluster labels of the points
    # returns: None, but shows a scatter plot with different colors for each label and the center of each cluster

    # get the number of unique labels
    num_labels = len(np.unique(labels))

    # generate a list of distinct colors
    colors = plt.cm.rainbow(np.linspace(0, 1, num_labels))

    # create a new figure
    plt.figure(figsize=figsize)

    # loop through the labels and plot each cluster
    for i, label in enumerate(np.unique(labels)):
        # get the points that belong to this cluster
        cluster_points = data[labels == label]

        # get the mean of this cluster, which is also the center
        cluster_mean = cluster_points.mean(axis=0)

        # plot the points with the corresponding color
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], alpha=0.5, s=10)

        # plot the center with a bigger marker and a text label
        plt.scatter(cluster_mean[0], cluster_mean[1], color=colors[i], marker='*', s=200)
        plt.text(cluster_mean[0], cluster_mean[1], str(class_dict[int(label)]) if class_dict is not None else str(label), fontsize=16, color='black')

    # set the axis labels and show the plot
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.show()