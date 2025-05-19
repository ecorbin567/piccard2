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

def clustering_prep(network_table, id):
    '''
    Converts a piccard network table into a 3d numpy array of all possible paths and their corresponding
    features. This will be used for clustering with tscluster.
    Note that you must run pc.create_network_table() before this function.

    Returns a tuple of a 3d numpy array and a corresponding dictionary of labels showing
    the shape of the array.
    '''
    # Find all years present in the data. These will be used as timesteps for tscluster.
    col_list = network_table.columns.to_list()
    year_cols = [col for col in col_list if id in col]
    years = [col[-4:] for col in year_cols]

    # Only add features for which there are variables in every year. Otherwise the shape of
    # the 3D array used for tscluster will not make sense.
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

def cluster(network_table, G, id, num_clusters, scheme):
    '''
    Runs one of tscluster's clustering algorithms and adds the resulting cluster assignments to the
    nodes as an additional feature.
    '''
    # Get the data into the correct format. See the documentation for clustering_prep
    arr, label_dict = clustering_prep(network_table, id)
    
    # Initialize the model
    opt_ts = OptTSCluster(
        n_clusters=num_clusters,
        scheme=scheme,
        n_allow_assignment_change=None # Allow as many changes as possible
    )
    # Assign clusters
    opt_ts.fit(arr, label_dict=label_dict)
