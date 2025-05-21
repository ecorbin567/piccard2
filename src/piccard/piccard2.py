import warnings
import pandas as pd
import geopandas as gpd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
import re
import piccard as pc

from tscluster.opttscluster import OptTSCluster
# from tscluster.tskmeans import TSKmeans, TSGlobalKmeans
from tscluster.preprocessing.utils import load_data
# from tscluster.metrics import inertia, max_dist
# from tscluster.tsplot import tsplot

def clustering_prep(network_table, id, cols=[]):
    '''
    Converts a piccard network table into a 3d numpy array of all possible paths and their corresponding
    features. This will be used for clustering with tscluster.
    The user can (optionally) input a list of columns that they want to be considered in the clustering algorithm, 
    and the function will check that these columns are valid.

    Note that you must run pc.create_network_table() before this function.

    Returns a tuple of a 3d numpy array and a corresponding dictionary of labels showing
    the shape of the array.
    '''
    # default to considering all features
    if cols == []:
        cols = network_table.columns.to_list()

    # Find all years present in the data. These will be used as timesteps for tscluster.
    year_cols = [col for col in network_table.columns.to_list() if id in col]
    years = [col[-4:] for col in year_cols]

    # Only add features that are numerical or nan. the user should have selected accordingly
    # but this is a sanity check
    col_list = []

    for col in cols:
        if col in network_table.columns.to_list():
            non_numerical_val_in_col = False
            for entry in network_table[col]:
                if isinstance(entry, str) and '_' in entry: # make sure underscores don't get converted to numbers
                    non_numerical_val_in_col = True
                    break
                try:
                    int(entry)  # see if it is either an int or an int masquerading as a string
                except ValueError:
                    try:
                        float(entry)  # see if it is either a float or a float masquerading as a string
                    except ValueError:
                        if entry != 'NaN' and entry != 'nan': # see if it is nan
                            non_numerical_val_in_col = True
                            break
            if not non_numerical_val_in_col:
                col_list.append(col)

    # Only add features for which there are variables in every year. Otherwise the shape of
    # the 3D array used for tscluster will not make sense.
    # note: we can improve on this with some version of the ppandas library (https://link.springer.com/article/10.1007/s10618-024-01054-7)
    cols_in_every_year = []
    add_to_list = True
    col_names_without_year = list({col[:-4] for col in col_list})
    for col in col_names_without_year:
        add_to_list = True
        for year in years:
            if f"{col}{year}" not in col_list:
                add_to_list = False
                break
        for year in years:
            if add_to_list:
                cols_in_every_year.append(f"{col}{year}")

    # Extract features for each year and add them to a 2D array representing that year. 
    # Then add that array to a list of arrays representing the 3D array used for tscluster.
    list_of_arrays = []
    for year in years:
        year_statistics = network_table[[col for col in cols_in_every_year if year in col]].to_numpy()
        list_of_arrays.append(year_statistics)
    
    # Run load_data and return the resulting tuple of a 3d numpy array and its corresponding
    # label dictionary. This can then be preprocessed using tscluster's scalers.
    return load_data(list_of_arrays)

def cluster(network_table, G, id, num_clusters, scheme, arr=None, label_dict=None):
    '''
    Runs one of tscluster's clustering algorithms and adds the resulting cluster assignments to the
    nodes as an additional feature.
    Users can choose to only input the network table, in which case clustering_prep will be run for them with the default columns,
    or they can choose to run clustering_prep on their own and then have the option to apply one or both of the
    normalization methods available in tscluster.preprocessing.utils.
    '''
    # Get the data into the correct format. See the documentation for clustering_prep
    if arr is None and label_dict is None:
        arr, label_dict = clustering_prep(network_table, id)
    
    # Initialize the model
    opt_ts = OptTSCluster(
        n_clusters=num_clusters,
        scheme=scheme,
        n_allow_assignment_change=None # Allow as many changes as possible
    )
    # Assign clusters
    opt_ts.fit(arr, label_dict=label_dict)
