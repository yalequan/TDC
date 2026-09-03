#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
DIF Model Utilities
Functions for training, evaluating, and applying DIF detection models.

Blinded code for peer review.


@author: ________
"""


# Auto-check and optionally install required packages
import sys
import subprocess
import importlib.util


def _check_package_installed(import_name):
    """Check if a package is installed.
    
    Args:
        import_name (str): The name to use when importing.
        
    Returns:
        bool: True if installed, False otherwise.
    """
    return importlib.util.find_spec(import_name) is not None


def _install_packages(packages_to_install):
    """Install a list of packages.
    
    Args:
        packages_to_install (list of tuple): List of (package_name, import_name) tuples to install.
    """
    print("\nInstalling packages...")
    for package_name, import_name in packages_to_install:
        print(f"  Installing '{package_name}'...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
            print(f"  Successfully installed '{package_name}'")
        except subprocess.CalledProcessError as e:
            print(f"  Error installing '{package_name}': {e}")
            print(f"  Please install manually: pip install {package_name}")
            raise


# Define required packages with their install and import names
_REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("tensorflow", "tensorflow"),
    ("scikit-learn", "sklearn"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("scikit-multilearn", "skmultilearn"),
    ("networkx", "networkx"),
    ("scipy", "scipy"),
]

# Check for missing packages
missing_packages = []
for package_name, import_name in _REQUIRED_PACKAGES:
    if not _check_package_installed(import_name):
        missing_packages.append((package_name, import_name))

# If packages are missing, prompt user
if missing_packages:
    print("\n" + "="*70)
    print("InterDIFNet Dependencies Check")
    print("="*70)
    print("\nThe following packages are required but not installed:")
    for package_name, _ in missing_packages:
        print(f"  - {package_name}")
    
    print("You can install them by running:")
    print(f"  pip install {' '.join([pkg for pkg, _ in missing_packages])}")
    
    print("\nWould you like to install them automatically now? (y/n): ", end="", flush=True)
    
    try:
        response = input().strip().lower()
        if response in ['y', 'yes']:
            _install_packages(missing_packages)
            print("\nAll dependencies installed successfully!\n")
        else:
            print("\nPlease install the required packages before using InterDIFNet.")
            print(f"Run: pip install {' '.join([pkg for pkg, _ in missing_packages])}")
            sys.exit(1)
    except (EOFError, KeyboardInterrupt):
        print("\n\nInstallation cancelled. Please install required packages manually.")
        print(f"Run: pip install {' '.join([pkg for pkg, _ in missing_packages])}")
        sys.exit(1)
else:
    print("All TDC dependencies are installed.")

# Now import all required packages (suppressing linter warnings about import order)
import numpy as np # noqa: E402
import pandas as pd # noqa: E402
import glob # noqa: E402
import tensorflow as tf # noqa: E402
from tensorflow.keras import Input # noqa: E402
from tensorflow.keras.models import Sequential, Model # noqa: E402
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization # noqa: E402
from tensorflow.keras.optimizers import Adam # noqa: E402
from tensorflow.keras.callbacks import EarlyStopping # noqa: E402
from tensorflow.keras.regularizers import l2 # noqa: E402
import matplotlib.pyplot as plt # noqa: E402
from skmultilearn.model_selection import iterative_train_test_split # noqa: E402
import os # noqa: E402
import pickle # noqa: E402
import json # noqa: E402
import warnings # noqa: E402
from sklearn.preprocessing import StandardScaler # noqa: E402
from sklearn.metrics import silhouette_score # noqa: E402
from scipy.cluster.hierarchy import linkage, fcluster # noqa: E402
from scipy.spatial.distance import pdist # noqa: E402
from tensorflow.keras.metrics import AUC # noqa: E402
from sklearn.metrics import roc_curve # noqa: E402
from sklearn.utils import shuffle # noqa: E402
from itertools import combinations # noqa: E402
import seaborn as sns # noqa: E402
import tensorflow.keras.backend as K # noqa: E402
import gc # noqa: E402
import re # noqa: E402
import networkx as nx # noqa: E402
from pathlib import Path # noqa: E402

# Clustering DIF Item Functions
def load_dif_data(groups, n, perc, r, data_folder=None):

    dif_data = {'DIF_a': pd.DataFrame(), 'DIF_b': pd.DataFrame()}
    
    # Construct file path with optional folder
    filename = f"Classification_Results_{groups}_{n}_{perc}_Replication{r}.csv"
    if data_folder:
        filepath = os.path.join(data_folder, filename)
    else:
        filepath = filename
    
    df = pd.read_csv(filepath)
    
    # Check which DIF type this file contains
    dif_a_cols = [col for col in df.columns if col.startswith('DIF_a_')]
    dif_b_cols = [col for col in df.columns if col.startswith('DIF_b_')]
    
    if dif_a_cols:
        #print(f"  Found {len(dif_a_cols)} DIF_a columns")
        if dif_data['DIF_a'].empty:
            dif_data['DIF_a'] = df.copy()
        else:
            # Merge with existing DIF_a data
            dif_data['DIF_a'] = pd.concat([dif_data['DIF_a'], df], ignore_index=True)
    
    if dif_b_cols:
        #print(f"  Found {len(dif_b_cols)} DIF_b columns")
        if dif_data['DIF_b'].empty:
            dif_data['DIF_b'] = df.copy()
        else:
            # Merge with existing DIF_b data
            dif_data['DIF_b'] = pd.concat([dif_data['DIF_b'], df], ignore_index=True)
    
    if not dif_a_cols and not dif_b_cols:
        raise KeyError(f"Error: No DIF_a or DIF_b columns found in {filepath}")
    
    return dif_data

def extract_groups_from_columns(dif_data, dif_type):
    """
    Extract unique group names from column headers for specified DIF type
    """
    column_names = dif_data.columns
    groups = set()
    pattern = f'{dif_type}_Group(\\d+)Group(\\d+)'
    
    for col in column_names:
        matches = re.findall(pattern, col)
        for match in matches:
            groups.add(f'Group{match[0]}')
            groups.add(f'Group{match[1]}')
    
    return sorted(list(groups), key=lambda x: int(x.replace('Group', '')))

def create_dif_matrix_per_item(dif_data, groups, item_index, dif_type='DIF_a'):
    """
    Create DIF matrix from pairwise comparison data for a specific item
    
    Parameters:
        df: DataFrame containing DIF data
        groups: List of group names
        item_index: Row index for the specific item
        dif_type: 'DIF_a' or 'DIF_b'
    
    Returns:
        pd.DataFrame: DIF matrix for the specified item
    """
    
    # Check for missing data in the specific row
    item_row = dif_data.iloc[item_index]
    dif_cols = [col for col in dif_data.columns if col.startswith(f'{dif_type}_')]
    
    if item_row[dif_cols].isna().any() or item_row[dif_cols].map(np.isinf).any():
        print(f"Warning: Missing or infinite data found in item {item_index}")
    
    dif_matrix = pd.DataFrame(0.0, index=groups, columns=groups)
    
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            group1, group2 = groups[i], groups[j]
            
            # Check both possible column name orders
            col_name1 = f'{dif_type}_{group1}{group2}'
            col_name2 = f'{dif_type}_{group2}{group1}'
            
            if col_name1 in dif_data.columns:
                dif_value = item_row[col_name1]
            elif col_name2 in dif_data.columns:
                dif_value = item_row[col_name2]
            else:
                dif_value = np.nan
                
            dif_matrix.loc[group1, group2] = dif_value
            dif_matrix.loc[group2, group1] = dif_value
            
    return dif_matrix

def Floyd_Warshall_Closure(dif_matrix, threshold, verbose=False):
    """
    Enforce transitive closure on DIF matrix using Floyd-Warshall-like algorithm.
    If there's a path of similar groups (DIF < threshold), all pairs in that path 
    should be considered similar.
    
    Parameters:
        dif_matrix: pandas DataFrame with DIF values
        threshold: DIF threshold for similarity
        verbose: Whether to print detailed information about changes
    
    Returns:
        pandas DataFrame: Modified DIF matrix with transitive closure enforced
        dict: Information about changes made
    """
    # Work with a copy to avoid modifying original
    modified_matrix = dif_matrix.copy()
    groups = list(dif_matrix.index)
    n_groups = len(groups)
    
    changes_made = []
    
    if verbose:
        print(f"Enforcing transitive closure with threshold {threshold}")
        print(f"Initial similarity pairs (DIF < {threshold}):")
        initial_pairs = []
        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                if modified_matrix.iloc[i, j] < threshold:
                    initial_pairs.append((groups[i], groups[j], modified_matrix.iloc[i, j]))
        for g1, g2, dif_val in initial_pairs:
            print(f"  {g1} - {g2}: {dif_val:.4f}")
    
    # Use Floyd-Warshall-like algorithm for transitive closure
    # For each potential intermediate node k
    for k in range(n_groups):
        group_k = groups[k]
        
        # For each pair of nodes i, j
        for i in range(n_groups):
            for j in range(n_groups):
                if i == j or i == k or j == k:
                    continue
                
                group_i, group_j = groups[i], groups[j]
                
                # Get current DIF values
                dif_ik = modified_matrix.iloc[i, k]  # i to k
                dif_kj = modified_matrix.iloc[k, j]  # k to j
                dif_ij = modified_matrix.iloc[i, j]  # i to j (current)
                
                # Check if we have a path i~k~j where both edges are similar
                if (not np.isnan(dif_ik) and not np.isnan(dif_kj) and 
                    dif_ik < threshold and dif_kj < threshold):
                    
                    # Calculate new DIF value for i~j through path i~k~j
                    # Use maximum of the path (weakest link determines strength)
                    new_dif_ij = max(dif_ik, dif_kj)
                    
                    # If current i~j relationship is weaker (higher DIF) than the path,
                    # or if i~j was not similar before, update it
                    if (np.isnan(dif_ij) or dif_ij >= threshold or new_dif_ij < dif_ij):
                        
                        # Only make changes if it improves the relationship
                        if np.isnan(dif_ij) or new_dif_ij < dif_ij:
                            # Update both symmetric positions
                            modified_matrix.iloc[i, j] = new_dif_ij
                            modified_matrix.iloc[j, i] = new_dif_ij
                            
                            changes_made.append({
                                'type': 'transitive_closure',
                                'path': (group_i, group_k, group_j),
                                'original_dif_ij': dif_ij,
                                'new_dif_ij': new_dif_ij,
                                'dif_ik': dif_ik,
                                'dif_kj': dif_kj,
                                'reason': f"{group_i}~{group_k} ({dif_ik:.4f}) and {group_k}~{group_j} ({dif_kj:.4f}) → {group_i}~{group_j} ({new_dif_ij:.4f})"
                            })
                            
                            if verbose:
                                original_str = f"{dif_ij:.4f}" if not np.isnan(dif_ij) else "None"
                                print(f"  Path {group_i}→{group_k}→{group_j}: {group_i}~{group_j} updated from {original_str} to {new_dif_ij:.4f}")
    
    # Summary information
    closure_info = {
        'total_changes': len(changes_made),
        'changes_detail': changes_made,
        'final_similarity_pairs': []
    }
    
    # Collect final similarity pairs
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            if modified_matrix.iloc[i, j] < threshold:
                closure_info['final_similarity_pairs'].append(
                    (groups[i], groups[j], modified_matrix.iloc[i, j])
                )
    
    if verbose:
        print("\nTransitive closure completed")
        print(f"Total changes made: {closure_info['total_changes']}")
        print(f"Final similarity pairs (DIF < {threshold}):")
        for g1, g2, dif_val in closure_info['final_similarity_pairs']:
            print(f"  {g1} - {g2}: {dif_val:.4f}")
    
    return modified_matrix, closure_info

def connected_components_clustering(dif_matrix, groups, dif_threshold=0.50, 
                                    dif_type='DIF_a', verbose_closure=False):
    """
    Find connected components where groups have DIF below threshold.
    Groups in same component have SIMILAR parameters (low DIF probability).
    
    Parameters:
        dif_matrix: DIF probability matrix
        groups: List of group names
        dif_threshold: Probability threshold - groups with DIF prob BELOW this are connected
        dif_type: Type of DIF being analyzed
        verbose_closure: Whether to print closure details
    
    Returns:
        cluster_dict, connected_components, graph, edge_details, closure_info, original_edges, transitive_edges
    """
    # First, identify original edges before transitive closure
    original_edges = []
    original_edge_dict = {}  # Track original DIF values
    
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            dif_value = dif_matrix.iloc[i, j]
            if not np.isnan(dif_value) and dif_value < dif_threshold:
                edge_key = (min(groups[i], groups[j]), max(groups[i], groups[j]))
                original_edges.append((groups[i], groups[j], dif_value))
                original_edge_dict[edge_key] = dif_value
    
    # Apply transitive closure
    modified_matrix, closure_info = Floyd_Warshall_Closure(
        dif_matrix, dif_threshold, verbose=verbose_closure
    )
    
    # Identify transitive closure edges
    transitive_edges = []
    
    # Check all edges in the modified matrix
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            modified_dif_value = modified_matrix.iloc[i, j]
            
            if not np.isnan(modified_dif_value) and modified_dif_value < dif_threshold:
                edge_key = (min(groups[i], groups[j]), max(groups[i], groups[j]))
                
                # If this edge wasn't in the original edges, or if it was modified
                if edge_key not in original_edge_dict:
                    # This is a completely new edge created by transitive closure
                    transitive_edges.append((groups[i], groups[j], modified_dif_value))
                elif abs(original_edge_dict[edge_key] - modified_dif_value) > 1e-10:
                    # This edge was modified by transitive closure (but keep it as transitive)
                    transitive_edges.append((groups[i], groups[j], modified_dif_value))
                    # Remove from original edges since it was modified
                    original_edges = [(g1, g2, dif) for g1, g2, dif in original_edges 
                                    if (min(g1, g2), max(g1, g2)) != edge_key]
    
    # Create graph using the transitively closed matrix
    G = nx.Graph()
    G.add_nodes_from(groups)
    
    # Add all edges (original + transitive) to the graph
    edges_added = 0
    edge_details = []
    
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            dif_value = modified_matrix.iloc[i, j]
            # Connect groups if DIF probability is LOW (below threshold)
            if not np.isnan(dif_value) and dif_value < dif_threshold:
                G.add_edge(groups[i], groups[j], weight=1 - dif_value, dif=dif_value)
                edges_added += 1
                edge_details.append((groups[i], groups[j], dif_value))
    
    # Find connected components (groups that should share parameters)
    connected_components = list(nx.connected_components(G))
    
    # Create cluster assignments
    cluster_dict = {}
    for cluster_id, component in enumerate(connected_components, 1):
        for group in component:
            cluster_dict[group] = cluster_id
    
    # Handle isolated nodes (groups that don't cluster with any other)
    # These are groups with HIGH DIF probability that should have unique parameters
    isolated_count = 0
    for group in groups:
        if group not in cluster_dict:
            cluster_dict[group] = len(connected_components) + 1 + isolated_count
            isolated_count += 1
    
    return cluster_dict, connected_components, G, edge_details, closure_info, original_edges, transitive_edges

def visualize_dif_matrix_per_item(dif_matrix, dif_type, item_index):
    """Visualize DIF matrix as heatmap for a specific item"""
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(dif_matrix, dtype=bool))
        
    sns.heatmap(dif_matrix, mask=mask, annot=True, cmap=("PiYG"), 
               fmt='.4f', square=True, linewidths=0.5, 
               cbar_kws={'label': 'Pairwise DIF Probability'},
               vmin=0, vmax=1, center=0.50)
    
    if dif_type == 'DIF_a':
        plt.title(f'DIF on a - Item {item_index + 1} Pairwise Probability Matrix')
    elif dif_type == 'DIF_b':
        plt.title(f'DIF on b - Item {item_index + 1} Pairwise Probability Matrix')
    plt.tight_layout()
    plt.show()


def visualize_connected_components_per_item(G, cluster_dict, groups,
                                          dif_type='DIF_a', item_index=0, save_plot=False, 
                                          n=None, p=None, r=None, groups_name=None,
                                          original_edges=None, transitive_edges=None, verbose_edges=False,
                                          show_title=True, results_folder=None, color_transitive=False,
                                          show_plots=True, legend_fontsize=12):
    """
    Visualize the graph with clusters colored for a specific item, showing original vs transitive closure edges.
    
    Parameters:
    -----------
    G : networkx.Graph
        Graph object with nodes and edges
    cluster_dict : dict
        Dictionary mapping groups to cluster IDs
    groups : list
        List of group names
    dif_type : str
        'DIF_a' or 'DIF_b'
    item_index : int
        Item index (0-based)
    save_plot : bool
        Whether to save the plot to file
    n, p, r : int, optional
        Sample size, DIF percentage, replication number (for filename)
    groups_name : str, optional
        Group name for filename
    original_edges : list, optional
        List of original edges (drawn as solid black lines)
    transitive_edges : list, optional
        List of transitive closure edges (drawn as dashed red lines)
    verbose_edges : bool
        Whether to print edge classification details
    show_title : bool
        Whether to display plot title (default: True)
    color_transitive : bool
        Whether to color-code transitive vs non-transitive edges and show legend (default: False)
    show_plots : bool
        Whether to display plots interactively. If False, plots are only saved to files.
        Default: True
    legend_fontsize : int
        Font size for the legend.
        Default: 12
    """
    # Disable interactive mode if not showing plots to prevent IDE from displaying
    if not show_plots:
        plt.ioff()
    
    plt.figure(figsize=(12, 8))
    
    # Create layout
    try:
        pos = nx.spring_layout(G, seed=42, k=3, iterations=100)
    except:
        pos = nx.circular_layout(G)
    
    # Get unique clusters and assign colors
    unique_clusters = sorted(set(cluster_dict.values()))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
    
    # Draw nodes colored by cluster
    for cluster_id in unique_clusters:
        cluster_nodes = [g for g in groups if cluster_dict[g] == cluster_id]
        nx.draw_networkx_nodes(G, pos, nodelist=cluster_nodes, 
                             node_color=[colors[cluster_id-1]], 
                             node_size=3000, alpha=0.8,
                             edgecolors='black', linewidths=1)
    
    # Debug output for edge classification
    if verbose_edges:
        print(f"\n{dif_type} Item {item_index + 1} - Edge Classification:")
        print(f"Original edges ({len(original_edges) if original_edges else 0}):")
        if original_edges:
            for edge in original_edges:
                print(f"  {edge[0]} - {edge[1]}: {edge[2]:.4f} (original)")
        print(f"Transitive edges ({len(transitive_edges) if transitive_edges else 0}):")
        if transitive_edges:
            for edge in transitive_edges:
                print(f"  {edge[0]} - {edge[1]}: {edge[2]:.4f} (transitive)")
    
    # Draw edges based on color_transitive setting
    if color_transitive:
        # Original behavior: different colors for different edge types
        # Draw original edges (solid black lines)
        if original_edges:
            original_edge_list = [(edge[0], edge[1]) for edge in original_edges]
            nx.draw_networkx_edges(G, pos, edgelist=original_edge_list, 
                                 edge_color='black', width=2, alpha=1, style='solid')
        
        # Draw transitive closure edges (thick dashed red lines)
        if transitive_edges:
            transitive_edge_list = [(edge[0], edge[1]) for edge in transitive_edges]
            nx.draw_networkx_edges(G, pos, edgelist=transitive_edge_list, 
                                 edge_color='red', width=4, alpha=0.8, style='dashed')
    else:
        # Simplified behavior: all edges same color, no distinction
        all_edges = []
        if original_edges:
            all_edges.extend([(edge[0], edge[1]) for edge in original_edges])
        if transitive_edges:
            all_edges.extend([(edge[0], edge[1]) for edge in transitive_edges])
        
        if all_edges:
            nx.draw_networkx_edges(G, pos, edgelist=all_edges, 
                                 edge_color='black', width=2, alpha=1, style='solid')
    
    # If no edge separation provided, draw all edges as before (fallback)
    if not original_edges and not transitive_edges:
        nx.draw_networkx_edges(G, pos, alpha=1, edge_color='black', width=2)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Uncomment to Add edge labels with DIF values
    # edge_labels = {}
    # for u, v, d in G.edges(data=True):
    #     if 'dif' in d:
    #         edge_labels[(u, v)] = f"{d['dif']:.3f}"
    
    #nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    
    # Add legend for edge types only if color_transitive is True
    if color_transitive:
        if original_edges and transitive_edges:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='black', linewidth=2, label='Direct Measurement Invariance Relationship'),
                Line2D([0], [0], color='red', linewidth=4, linestyle='--', label='Transitive closure connections')
            ]
            # Place legend at bottom center, outside the plot area
            plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=legend_fontsize)
        elif transitive_edges and not original_edges:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='red', linewidth=4, linestyle='--', label='Transitive closure connections')
            ]
            # Place legend at bottom center, outside the plot area
            plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.02), fontsize=legend_fontsize)
        elif original_edges and not transitive_edges:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='black', linewidth=2, label='Direct Measurement Invariance Relationship')
            ]
            # Place legend at bottom center, outside the plot area
            plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.02), fontsize=legend_fontsize)
    
    # Add title only if show_title is True
    if show_title:
        if dif_type == "DIF_b":
            plt.title(f'{groups_name} Groups TDC Plot for Item {item_index + 1} \nClustering for DIF on b', 
                      fontsize=14, fontweight='bold')
        elif dif_type == "DIF_a":
            plt.title(f'{groups_name} Groups TDC Plot for Item {item_index + 1} \nClustering for DIF on a', 
                   fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot and n is not None and p is not None and r is not None and groups_name is not None:
        # Create sensible filename
        filename = f"DIF_Clustering_Components_{groups_name}Groups_{dif_type}_N{n}_P{p}_R{r}_Item{item_index+1}.png"
        
        # Construct full file path with optional folder
        if results_folder:
            filepath = os.path.join(results_folder, filename)
        else:
            filepath = filename
            
        plt.savefig(filepath, dpi=720, bbox_inches='tight')
        print(f"    Plot saved: {filepath}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Restore interactive mode
    plt.ion()


def test_transitive_closure_visualization(groups, n, p, r, dif_type='DIF_a', item_index=0, verbose=True):
    """
    Test function to verify transitive closure visualization is working correctly.
    """
    print(f"Testing transitive closure visualization for {groups} groups, N={n}, P={p}, R={r}")
    print(f"DIF type: {dif_type}, Item: {item_index + 1}")
    
    # Load data
    dif_data = load_dif_data(groups, n, p, r)
    df = dif_data[dif_type]
    
    if df.empty:
        print(f"No data available for {dif_type}")
        return
    
    # Extract groups
    extracted_groups = extract_groups_from_columns(df, dif_type)
    print(f"Groups found: {extracted_groups}")
    
    # Create DIF matrix
    dif_matrix = create_dif_matrix_per_item(df, extracted_groups, item_index, dif_type)
    print(f"\nOriginal DIF matrix for item {item_index + 1}:")
    print(dif_matrix)
    
    # Test with a threshold that should create some transitive connections
    threshold = 0.3  # You can adjust this
    print(f"\nUsing threshold: {threshold}")
    
    # Run the clustering with verbose closure
    cluster_dict, components, G, edges, closure_info, original_edges, transitive_edges = connected_components_clustering(
        dif_matrix, extracted_groups, threshold, dif_type, verbose_closure=True
    )
    
    print(f"\nClosure info: {closure_info['total_changes']} changes made")
    
    # Show edge classification
    print(f"\nEdge Classification:")
    print(f"Original edges: {len(original_edges)}")
    for edge in original_edges:
        print(f"  {edge[0]} - {edge[1]}: {edge[2]:.4f} (BLACK - original)")
    
    print(f"Transitive edges: {len(transitive_edges)}")
    for edge in transitive_edges:
        print(f"  {edge[0]} - {edge[1]}: {edge[2]:.4f} (RED - transitive)")
    
    # Create visualization with verbose edge info
    visualize_connected_components_per_item(
        G, cluster_dict, extracted_groups, dif_type, item_index,
        original_edges=original_edges, transitive_edges=transitive_edges, verbose_edges=True
    )


def analyze_item_threshold(dif_matrix, groups, threshold, dif_type='DIF_a', item_index=0, verbose_closure=False):
    """Analyze a single threshold value for a specific item and return results"""
    cluster_dict, components, G, edges, closure_info, original_edges, transitive_edges = connected_components_clustering(
        dif_matrix, groups, threshold, dif_type, verbose_closure)
    
    # Create results DataFrame
    cluster_df = pd.DataFrame({
        'Group': groups, 
        'Cluster': [cluster_dict[g] for g in groups]
    }).sort_values(by='Cluster')
    
    # Calculate statistics
    num_clusters = len(set(cluster_dict.values()))
    num_edges = G.number_of_edges()
    cluster_sizes = cluster_df['Cluster'].value_counts().sort_index()
    multi_group_clusters = cluster_sizes[cluster_sizes > 1]
    
    # Debugging Code
    '''
    Prints the DIF matrix before and after transitive closure. Lists the
    changes and the number of chnages made.
    '''
    # print(f"Before transitive closure: {dif_matrix.values}")
    # modified_matrix, closure_info = Floyd_Warshall_Closure(dif_matrix, threshold, verbose=True)
    # print(f"After transitive closure: {modified_matrix.values}")
    # print(f"Changes made: {closure_info['total_changes']}")
    
    return {
        'item_index': item_index,
        'threshold': threshold,
        'cluster_df': cluster_df,
        'components': components,
        'graph': G,
        'edges': edges,
        'closure_info': closure_info,
        'num_clusters': num_clusters,
        'num_edges': num_edges,
        'cluster_sizes': cluster_sizes,
        'multi_group_clusters': multi_group_clusters
    }

def find_recommended_threshold_per_item(dif_matrix, groups, dif_type='DIF_a', 
                                      item_index=0, test_thresholds=None, verbose_closure=False):
    """
    Find the recommended threshold for a specific item
    """
    if test_thresholds is None:
        # Default threshold range
        test_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
        #test_thresholds = [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    
    all_results = {}
    for threshold in test_thresholds:
        result = analyze_item_threshold(dif_matrix, groups, threshold, dif_type, item_index, verbose_closure)
        all_results[threshold] = result
    
    # Create summary comparison
    summary_df = pd.DataFrame([
        {
            'Threshold': threshold,
            'Num_Clusters': result['num_clusters'],
            'Num_Edges': result['num_edges'],
            'Multi_Group_Clusters': len(result['multi_group_clusters']),
            'Largest_Cluster_Size': result['cluster_sizes'].max() if len(result['cluster_sizes']) > 0 else 0
        }
        for threshold, result in all_results.items()
    ])
    
    # Find recommended threshold
    meaningful_thresholds = summary_df[summary_df['Multi_Group_Clusters'] > 0]
    
    recommended_threshold = None
    if len(meaningful_thresholds) > 0:
        # Choose the smallest threshold that creates multi-group clusters
        recommended_threshold = meaningful_thresholds.iloc[0]['Threshold']
    else:
        # If no threshold creates clusters, use a middle value for visualization
        recommended_threshold = test_thresholds[len(test_thresholds)//2]
    
    return summary_df, recommended_threshold, all_results


def display_item_results(dif_matrix, groups, threshold, dif_type='DIF_a', item_index=0, verbose_closure=False):
    """Display detailed results for a specific item and threshold"""
    cluster_dict, components, G, edges, closure_info, original_edges, transitive_edges = connected_components_clustering(
        dif_matrix, groups, threshold, dif_type, verbose_closure)
    
    # Create results DataFrame
    cluster_df = pd.DataFrame({
        'Group': groups, 
        'Cluster': [cluster_dict[g] for g in groups]
    }).sort_values(by='Cluster')
    
    # Calculate statistics
    num_clusters = len(set(cluster_dict.values()))
    num_edges = G.number_of_edges()
    cluster_sizes = cluster_df['Cluster'].value_counts().sort_index()
    multi_group_clusters = cluster_sizes[cluster_sizes > 1]
    
    if verbose_closure:
        print(f"\n{'='*60}")
        print(f"ITEM {item_index + 1} - {dif_type.upper()} RESULTS (Threshold: {threshold})")
        print(f"{'='*60}")
        print(f"Total groups: {len(groups)}")
        print(f"Number of clusters: {num_clusters}")
        print(f"Number of edges (connections): {num_edges}")
        
       # Display transitive closure information
        if closure_info['total_changes'] > 0:
            print("\nTransitive Closure Applied:")
        print(f"  Iterations: {closure_info['iterations']}")
        print(f"  Changes made: {closure_info['total_changes']}")
        if not verbose_closure:
            print("  (Use verbose_closure=True to see detailed changes)")
    else:
        print("\nTransitive Closure: No changes needed (already transitive)")
    
    if edges:
        print(f"\nFinal Connections (DIF < {threshold}):")
        for group1, group2, dif_val in sorted(edges, key=lambda x: x[2]):
            print(f"  {group1} - {group2}: {dif_type} = {dif_val:.4f}")
    
    # Generate visualizations
    visualize_connected_components_per_item(G, cluster_dict, groups, dif_type, item_index,
                                           original_edges=original_edges, transitive_edges=transitive_edges)
    
    return cluster_dict, components, G, edges, closure_info

def DIF_Cluster_Components_Per_Item(dif_data, dif_type, test_thresholds=None,
                                   show_matrices=False, items_to_analyze=None,
                                   verbose_closure=False, verbose = False):
    """
    Analyze DIF clustering for each item individually.
    This modified version *does not* generate plots directly.
    It returns all the necessary data for external plotting.

    Parameters:
        dif_data: Dictionary containing DIF data
        dif_type: 'DIF_a' or 'DIF_b'
        test_thresholds: List of thresholds to test
        show_matrices: Whether to show heatmap matrices (still handled here, but could be moved out)
        items_to_analyze: List of item indices to analyze (None for all)
        verbose_closure: Whether to show detailed transitive closure changes
    
    Returns:
        dict: Complete analysis results including clustering data for plotting
    """
    df = dif_data[dif_type]

    if df.empty:
        print(f"\nNo data available for {dif_type}")
        return None

    # Extract groups
    groups = extract_groups_from_columns(df, dif_type)
    
    if verbose:
        print(f"Found {len(groups)} groups: {groups}")
        print(f"Total items to analyze: {len(df)}")

    # Determine which items to analyze
    if items_to_analyze is None:
        items_to_analyze = list(range(len(df)))

    all_item_results = {}

    # Analyze each item
    for item_idx in items_to_analyze:
        if verbose:
            print(f"\n{'-'*50}")
            print(f"ANALYZING ITEM {item_idx + 1}")
            print(f"{'-'*50}")

        # Create DIF matrix for this specific item
        dif_matrix = create_dif_matrix_per_item(df, groups, item_idx, dif_type)

        # Show matrix if requested (this can stay here as it's a specific matrix view)
        if show_matrices:
            visualize_dif_matrix_per_item(dif_matrix, dif_type, item_idx)

        # Find recommended threshold for this item
        summary_df, recommended_threshold, threshold_results = find_recommended_threshold_per_item(
            dif_matrix, groups, dif_type, item_idx, test_thresholds, verbose_closure)

        # Display threshold summary
        if verbose:
            print(f"\nThreshold Summary for Item {item_idx + 1}:")
            print(summary_df.to_string(index=False))

        # Perform the clustering with the recommended threshold
        cluster_dict, components, G, edges, closure_info, original_edges, transitive_edges = connected_components_clustering(
            dif_matrix, groups, recommended_threshold, dif_type, verbose_closure)

        # Store results, including data needed for later plotting
        all_item_results[item_idx] = {
            'dif_matrix': dif_matrix,
            'summary_df': summary_df,
            'recommended_threshold': recommended_threshold,
            'threshold_results': threshold_results,
            'final_clustering': {
                'cluster_dict': cluster_dict,
                'components': components,
                'graph': G,
                'edges': edges,
                'closure_info': closure_info,
                'original_edges': original_edges,
                'transitive_edges': transitive_edges
            },
            'groups': groups, # Add groups for plotting outside
            'dif_type': dif_type, # Add dif_type for plotting outside
            'item_index': item_idx # Add item_index for plotting outside
        }

    return {
        'groups': groups,
        'dif_type': dif_type,
        'items_analyzed': items_to_analyze,
        'item_results': all_item_results
    }

def create_summary_across_items(results, dif_type, show_low_dif=True, show_high_dif=False):
    """
    Create a summary showing clustering patterns across all items
    
    Parameters:
        results: Results from DIF_Cluster_Components_Per_Item
        dif_type: 'DIF_a' or 'DIF_b'
        show_low_dif: Whether to show items with low DIF probabilities (default: True)
        show_high_dif: Whether to show items with high DIF probabilities (default: False)
    """
    if results is None:
        return
    
    groups = results['groups']
    item_results = results['item_results']
    
    # Create summary table
    summary_data = []
    for item_idx, item_result in item_results.items():
        final_clustering = item_result['final_clustering']
        cluster_dict = final_clustering['cluster_dict']
        
        # Count clusters and connections
        num_clusters = len(set(cluster_dict.values()))
        num_connections = final_clustering['graph'].number_of_edges()
        
        # Find multi-group clusters
        cluster_sizes = pd.Series([cluster_dict[g] for g in groups]).value_counts()
        multi_group_clusters = cluster_sizes[cluster_sizes > 1]
        
        summary_data.append({
            'Item': item_idx,
            'Recommended_Threshold': item_result['recommended_threshold'],
            'Num_Clusters': num_clusters,
            'Num_Connections': num_connections,
            'Multi_Group_Clusters': len(multi_group_clusters),
            'Connected_Groups': ', '.join([f"{u}-{v}" for u, v, _ in final_clustering['edges']])
        })
    
    # Analyze connections grouped by item (LOW DIF - below threshold)
    items_with_connections = []
    items_with_high_dif = []  # NEW: for high DIF probabilities
    
    for item_idx, item_result in item_results.items():
        edges = item_result['final_clustering']['edges']
        dif_matrix = item_result['dif_matrix']
        threshold = item_result['recommended_threshold']
        
        # LOW DIF: Items that have connections (existing logic)
        if edges:
            item_connections = []
            for group1, group2, dif_val in edges:
                pair_key = tuple(sorted([group1, group2]))
                item_connections.append((pair_key, dif_val))
            items_with_connections.append((item_idx, item_connections))
        
        # NEW: HIGH DIF - pairs exceeding threshold
        high_dif_pairs = []
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                group1, group2 = groups[i], groups[j]
                dif_val = dif_matrix.iloc[i, j]
                
                # Check if DIF value exists and exceeds threshold
                if not np.isnan(dif_val) and dif_val >= threshold:
                    pair_key = tuple(sorted([group1, group2]))
                    high_dif_pairs.append((pair_key, dif_val))
        
        if high_dif_pairs:
            items_with_high_dif.append((item_idx, high_dif_pairs))
    
    # Print LOW DIF probabilities (controlled by toggle)
    if show_low_dif and items_with_connections:
        if dif_type == "DIF_a": 
            print(f"\n{'#'*60}")
            print("Items With Low DIF Probabilities of DIF On a Across Groups:")
            print(f"{'#'*60}")
        
        elif dif_type == "DIF_b": 
            print(f"\n{'#'*60}")
            print("Items With Low DIF Probabilities of DIF On b Across Groups:")
            print(f"{'#'*60}")
            
        # Sort items by number of connections (descending)
        items_with_connections.sort(key=lambda x: len(x[1]), reverse=True)
        
        for item_idx, connections in items_with_connections:
            # Sort connections by DIF value (ascending - lowest DIF first)
            connections_sorted = sorted(connections, key=lambda x: x[1])
            
            print(f"Item {item_idx + 1}: {len(connections)} connection(s)")
            for (group1, group2), dif_val in connections_sorted:
                print(f"  {group1} - {group2} = {dif_val:.4f}")
    elif show_low_dif:
        print("\nNo low DIF connections found across any items.")
    
    # Print HIGH DIF probabilities (controlled by toggle)
    if show_high_dif and items_with_high_dif:
        if dif_type == "DIF_a": 
            print(f"\n{'#'*60}")
            print("Items With High DIF Probabilities of DIF On a Across Groups:")
            print(f"{'#'*60}")
        
        elif dif_type == "DIF_b": 
            print(f"\n{'#'*60}")
            print("Items With High DIF Probabilities of DIF On b Across Groups:")
            print(f"{'#'*60}")
            
        # Sort items by number of high DIF pairs (descending)
        items_with_high_dif.sort(key=lambda x: len(x[1]), reverse=True)
        
        for item_idx, high_dif_pairs in items_with_high_dif:
            # Sort pairs by DIF value (descending - highest DIF first)
            pairs_sorted = sorted(high_dif_pairs, key=lambda x: x[1], reverse=True)
            
            print(f"Item {item_idx + 1}: {len(high_dif_pairs)} high DIF pair(s)")
            for (group1, group2), dif_val in pairs_sorted:
                print(f"  {group1} - {group2} = {dif_val:.4f}")
    elif show_high_dif:
        print("\nNo high DIF probabilities found across any items.")

def extract_connected_groups_simple(dif_data, items_to_analyze=None, test_thresholds=None):
    """
    Extract connected groups for each item with minimal output.
    """
    results = {}
    
    # Unpack the thresholds if they are provided as a single list
    if isinstance(test_thresholds, (list, tuple)) and len(test_thresholds) == 2:
        opt_thr_a, opt_thr_b = test_thresholds
    else:
        opt_thr_a, opt_thr_b = None, None

    for dif_type in ['DIF_a', 'DIF_b']:
        df = dif_data[dif_type]
        
        if df.empty:
            continue
            
        groups = extract_groups_from_columns(df, dif_type)
        
        if items_to_analyze is None:
            items_to_analyze_local = list(range(len(df)))
        else:
            items_to_analyze_local = items_to_analyze
        
        results[dif_type] = {
            'groups': groups,
            'items': {}
        }
        
        # Set the correct threshold based on the DIF type
        if dif_type == 'DIF_a':
            current_threshold = opt_thr_a
        else:
            current_threshold = opt_thr_b

        for item_idx in items_to_analyze_local:
            dif_matrix = create_dif_matrix_per_item(df, groups, item_idx, dif_type)
            
            # Use the specified threshold directly if available
            if current_threshold is not None:
                final_threshold = current_threshold
            else:
                # Fallback to finding the recommended threshold if none is provided
                summary_df, final_threshold, all_results = find_recommended_threshold_per_item(
                    dif_matrix, groups, dif_type, item_idx, test_thresholds=None, verbose_closure=False
                )
            
            cluster_dict, components, G, edges, closure_info, original_edges, transitive_edges = connected_components_clustering(
                dif_matrix, groups, final_threshold, dif_type, verbose_closure=False)
            
            results[dif_type]['items'][item_idx] = {
                'threshold': final_threshold,
                'connected_components': list(components),
                'edges': edges,
                'cluster_dict': cluster_dict
            }
    
    return results

def print_connected_groups_simple(results):
    """
    Print only the connected groups for each item in a clean format.
    
    Parameters:
        results: Results from extract_connected_groups_simple()
    """
    for dif_type in ['DIF_a', 'DIF_b']:
        if dif_type not in results:
            continue
            
        print(f"\n{'='*60}")
        print(f"CONNECTED GROUPS - {dif_type.upper()}")
        print(f"{'='*60}")
        
        groups = results[dif_type]['groups']
        items = results[dif_type]['items']
        
        for item_idx in sorted(items.keys()):
            item_data = items[item_idx]
            components = item_data['connected_components']
            threshold = item_data['threshold']
            
            print(f"\nItem {item_idx + 1} (threshold: {threshold}):")
            
            if len(components) == 1 and len(components[0]) == len(groups):
                print("  All groups connected")
            elif len(components) == len(groups):
                print("  No groups connected (all isolated)")
            else:
                for i, component in enumerate(components):
                    if len(component) > 1:
                        component_list = sorted(list(component), key=lambda x: int(x.replace('Group', '')))
                        print(f"  Connected: {' ~ '.join(component_list)}")


# def generate_mirt_constraints(connected_results):
#     """
#     Generate MIRT constraints based on clustering results.
#     Low DIF probability -> Parameters should be constrained (similar)
#     """
#     constraints = {}
    
#     # Process both DIF types
#     for dif_type in ['DIF_a', 'DIF_b']:
#         if dif_type not in connected_results:
#             continue
            
#         items_data = connected_results[dif_type]['items']
        
#         # For each item
#         for item_idx, item_data in items_data.items():
#             if item_idx not in constraints:
#                 constraints[item_idx] = {'a1_constraints': [], 'd_constraints': []}
            
#             # Get connected components (groups that should share parameters)
#             components = item_data['connected_components']
            
#             # For each component with multiple groups
#             for component in components:
#                 if len(component) > 1:
#                     # Sort groups to ensure consistent ordering
#                     groups_in_component = sorted(list(component), key=lambda x: int(x.replace('Group', '')))
#                     group_nums = [int(g.replace('Group', '')) for g in groups_in_component]
                    
#                     # Create constraint string
#                     groups_str = ', '.join(map(str, group_nums))
                    
#                     # Add constraint based on DIF type
#                     if dif_type == 'DIF_a':
#                         constraints[item_idx]['a1_constraints'].append(f"[{groups_str}]")
#                     else:  # DIF_b
#                         constraints[item_idx]['d_constraints'].append(f"[{groups_str}]")
    
#     return constraints

def generate_mirt_constraints(results):
    """
    Generate MIRT constraint syntax for connected groups using INCLUSIVE syntax.
    Connected groups have no DIF, and should have equal parameters and be constrained together.
    """
    mirt_output = {}
    
    # Get all items that were analyzed
    all_items = set()
    for dif_type in ['DIF_a', 'DIF_b']:
        if dif_type in results:
            all_items.update(results[dif_type]['items'].keys())
    
    # Process each item
    for item_idx in sorted(all_items):
        mirt_output[item_idx] = {
            'a1_constraints': [],
            'd_constraints': []
        }
        
        # Process DIF_a (converts to a1 parameter)
        if 'DIF_a' in results and item_idx in results['DIF_a']['items']:
            dif_a_data = results['DIF_a']['items'][item_idx]
            groups = results['DIF_a']['groups']
            components = dif_a_data['connected_components']
            
            # Use a consistent approach for all cases
            for component in components:
                if len(component) > 1:
                    connected_groups = sorted(list(component), key=lambda x: int(x.replace('Group', '')))
                    group_ids = [int(g.replace('Group', '')) for g in connected_groups]
                    connected_str = ', '.join(map(str, group_ids))
                    mirt_output[item_idx]['a1_constraints'].append(f"CONSTRAINB[{connected_str}] = ({item_idx + 1}, a1)")
                
        # Process DIF_b (converts to d parameter)
        if 'DIF_b' in results and item_idx in results['DIF_b']['items']:
            dif_b_data = results['DIF_b']['items'][item_idx]
            groups = results['DIF_b']['groups']
            components = dif_b_data['connected_components']
            
            # Use a consistent approach for all cases
            for component in components:
                if len(component) > 1:
                    connected_groups = sorted(list(component), key=lambda x: int(x.replace('Group', '')))
                    group_ids = [int(g.replace('Group', '')) for g in connected_groups]
                    connected_str = ', '.join(map(str, group_ids))
                    mirt_output[item_idx]['d_constraints'].append(f"CONSTRAINB[{connected_str}] = ({item_idx + 1}, d)")
    
    return mirt_output

def format_mirt_constraint(constraint, item_number, param_type):
    """Formats a single MIRT constraint line."""
    if 'CONSTRAINB = ' in constraint:
        return f"CONSTRAINB = ({item_number}, {param_type})"
    
    match = re.search(r'\[(.*?)\]', constraint)
    if match:
        groups_str = match.group(1).strip()
        return f"CONSTRAINB[{groups_str}] = ({item_number}, {param_type})"
    return ""


def print_mirt_constraints(mirt_output):
    result = "F = 1-10\n"
    
    for item_idx in sorted(mirt_output.keys()):
        constraints = mirt_output[item_idx]
        item_number = item_idx + 1
        
        has_constraints = False
        constraint_lines = []

        # Process a1 constraints
        if constraints['a1_constraints']:
            for constraint in constraints['a1_constraints']:
                formatted_line = format_mirt_constraint(constraint, item_number, 'a1')
                if formatted_line:
                    constraint_lines.append(formatted_line)
                    has_constraints = True

        # Process d constraints
        if constraints['d_constraints']:
            for constraint in constraints['d_constraints']:
                formatted_line = format_mirt_constraint(constraint, item_number, 'd')
                if formatted_line:
                    constraint_lines.append(formatted_line)
                    has_constraints = True
        
        if has_constraints:
            result += f"# Group specific constraints for item {item_number}\n"
            for line in constraint_lines:
                result += line + "\n"
    
    result += "# Free latent means\n"
    result += "MEAN = F\n\n"
    
    return result

def save_mirt_constraints_to_file(mirt_constraints_string, filename="mirt_constraints.txt"):
    """
    Saves the generated MIRT constraint string to a specified text file.

    Args:
        mirt_constraints_string (str): The string containing the MIRT constraints.
        filename (str): The name of the file to save the constraints to.
    """
    try:
        with open(filename, 'w') as f:
            f.write(mirt_constraints_string)
    except IOError as e:
        print(f"Error saving MIRT constraints to file: {e}")

# Stan Constraints Generation and Export
def create_stan_constraints_from_clustering(connected_results, groups, n_items):
    """
    Convert clustering results to Stan constraint matrices.
    
    Parameters:
    -----------
    connected_results : dict
        Results from extract_connected_groups_simple()
    groups : list
        List of group names in order
    n_items : int
        Number of items
        
    Returns:
    --------
    dict containing constraint matrices and metadata
    """
    n_groups = len(groups)
    
    # Initialize constraint matrices (0 = free parameter)
    a_constraints = np.zeros((n_items, n_groups), dtype=int)
    b_constraints = np.zeros((n_items, n_groups), dtype=int)
    
    # Process discrimination parameters (DIF_a)
    a_constraint_counter = 0
    if 'DIF_a' in connected_results:
        a_constraints, a_constraint_counter = process_parameter_constraints(
            connected_results['DIF_a'], groups, n_items
        )
    
    # Process difficulty parameters (DIF_b)  
    b_constraint_counter = 0
    if 'DIF_b' in connected_results:
        b_constraints, b_constraint_counter = process_parameter_constraints(
            connected_results['DIF_b'], groups, n_items
        )
    
    return {
        'a_constraint_group': a_constraints,
        'b_constraint_group': b_constraints,
        'N_a_constraints': a_constraint_counter,
        'N_b_constraints': b_constraint_counter,
        'groups': groups,
        'n_items': n_items,
        'n_groups': n_groups
    }


def process_parameter_constraints(dif_results, groups, n_items):
    """
    Process constraints for a single parameter type (DIF_a or DIF_b).
    
    Parameters:
    -----------
    dif_results : dict
        DIF results for one parameter type
    groups : list
        List of group names
    n_items : int
        Number of items
        
    Returns:
    --------
    tuple: (constraint_matrix, constraint_counter)
    """
    n_groups = len(groups)
    constraint_matrix = np.zeros((n_items, n_groups), dtype=int)
    constraint_counter = 0
    
    for item_idx in range(n_items):
        if item_idx in dif_results['items']:
            components = dif_results['items'][item_idx]['connected_components']
            
            # Assign constraint IDs to multi-group clusters
            for component in components:
                if len(component) > 1:  # Only constrain if >1 group
                    constraint_counter += 1
                    
                    # Set constraint ID for all groups in this cluster
                    for group in component:
                        if group in groups:  # Safety check
                            group_idx = groups.index(group)
                            constraint_matrix[item_idx, group_idx] = constraint_counter
                # Single groups remain 0 (free)
    
    return constraint_matrix, constraint_counter


def apply_kmeans_to_dif_matrix(dif_matrix, groups, dif_type, item_index, random_state=12345, verbose=False):
    """
    Apply K-means clustering to a single DIF matrix using silhouette score for optimal cluster determination.
    
    This function extracts features from a DIF probability matrix and applies K-means clustering
    to identify groups with similar DIF patterns. The optimal number of clusters is determined
    automatically using the silhouette score.
    
    Parameters:
    -----------
    dif_matrix : pd.DataFrame
        DIF probability matrix for a single item, with groups as both rows and columns.
        Values represent pairwise DIF probabilities between groups.
    groups : list of str
        List of group names (e.g., ['Group1', 'Group2', 'Group3']).
    dif_type : str
        Type of DIF being analyzed ('DIF_a' for discrimination or 'DIF_b' for difficulty).
    item_index : int
        Zero-based item index being analyzed.
    random_state : int, optional
        Random seed for reproducibility. Should be unique per replication for proper variance.
        Default: 12345
    verbose : bool, optional
        Whether to print detailed clustering information.
        Default: False
        
    Returns:
    --------
    dict
        Clustering results containing:
        - 'cluster_dict': dict mapping group names to cluster IDs (1-indexed)
        - 'connected_components': list of sets, each set contains groups in a cluster
        - 'edges': list of tuples (group1, group2, dif_value) for groups in same cluster
        - 'features': array of original DIF features
        - 'features_scaled': array of standardized features
        - 'kmeans_model': fitted KMeans model object
        - 'cluster_centers': array of cluster centroids
        - 'silhouette_score': silhouette score of the clustering (-1 to 1)
        - 'inertia': within-cluster sum of squares
        - 'n_clusters': actual number of clusters found
        - 'requested_clusters': number of clusters requested
    
    Notes:
    ------
    - Uses StandardScaler to normalize features before clustering
    - Handles NaN values by replacing with 0.5 (neutral DIF probability)
    - Silhouette score of 0 indicates single cluster, -1 indicates degenerate cases
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    # Extract features from DIF matrix
    n_groups = len(groups)
    features = []
    
    for i, group in enumerate(groups):
        # For each group, create a feature vector of its DIF values with other groups
        group_features = []
        for j, other_group in enumerate(groups):
            if i != j:  # Exclude self-comparison
                dif_value = dif_matrix.loc[group, other_group]
                # Handle NaN values
                if pd.isna(dif_value):
                    dif_value = 0.5  # Use neutral DIF probability for missing values
                group_features.append(dif_value)
        features.append(group_features)
    
    features = np.array(features)
    
    # Standardize features (important for K-means)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Determine optimal number of clusters using silhouette score
    n_clusters = determine_optimal_k_silhouette(features_scaled, random_state=random_state)
    
    # Apply K-means clustering
    # Suppress sklearn convergence warning when clusters converge to fewer than requested
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Number of distinct clusters.*found smaller than n_clusters.*")
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
    
    # Check actual number of clusters found (K-means can converge to fewer clusters)
    actual_n_clusters = len(np.unique(cluster_labels))
    
    # Create cluster dictionary (same format as connected components method)
    cluster_dict = {}
    for i, group in enumerate(groups):
        cluster_dict[group] = cluster_labels[i] + 1  # +1 to match 1-indexed clusters
    
    # Calculate clustering quality metrics
    # Use actual_n_clusters to avoid silhouette_score error when only 1 cluster found
    if actual_n_clusters > 1 and actual_n_clusters < n_groups:
        silhouette_avg = silhouette_score(features_scaled, cluster_labels)
        inertia = kmeans.inertia_
    else:
        silhouette_avg = 0 if actual_n_clusters == 1 else -1  # -1 for degenerate cases
        inertia = kmeans.inertia_ if actual_n_clusters > 0 else 0
    
    # Create "edges" for compatibility with visualization
    # Groups in same cluster are considered "connected"
    edges = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if cluster_labels[i] == cluster_labels[j]:
                dif_value = dif_matrix.loc[groups[i], groups[j]]
                if pd.isna(dif_value):
                    dif_value = 0.5
                edges.append((groups[i], groups[j], dif_value))
    
    # Group clusters for connected components format
    connected_components = []
    unique_clusters = np.unique(cluster_labels)
    for cluster_id in unique_clusters:
        component = [groups[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
        connected_components.append(set(component))
    
    if verbose:
        print(f"K-means Clustering Results for {dif_type} - Item {item_index + 1}")
        print(f"Requested clusters: {n_clusters}, Actual clusters found: {actual_n_clusters}")
        print(f"Silhouette score: {silhouette_avg:.3f}")
        print(f"Inertia: {inertia:.3f}")
        if actual_n_clusters == 1:
            print("Note: All groups clustered together (no significant DIF detected)")
        print("Cluster assignments:")
        for group, cluster_id in cluster_dict.items():
            print(f"  {group}: Cluster {cluster_id}")
    
    return {
        'cluster_dict': cluster_dict,
        'connected_components': connected_components,
        'edges': edges,
        'features': features,
        'features_scaled': features_scaled,
        'kmeans_model': kmeans,
        'cluster_centers': kmeans.cluster_centers_,
        'silhouette_score': silhouette_avg,
        'inertia': inertia,
        'n_clusters': actual_n_clusters,  # Use actual clusters found
        'requested_clusters': n_clusters  # Keep track of what was requested
    }


def determine_optimal_k_silhouette(features, random_state=42):
    """
    Use silhouette score to determine optimal number of clusters.
    
    Tests k=2 to n_samples-1 and returns the k with the highest silhouette score.
    The silhouette score measures how similar an object is to its own cluster
    compared to other clusters.
    
    Parameters:
    -----------
    features : array-like
        Feature matrix for clustering (n_samples x n_features).
    random_state : int, optional
        Random seed for reproducible results within this selection process.
        Default: 42
        
    Returns:
    --------
    int
        Optimal number of clusters based on highest silhouette score.
        Returns 1 if n_samples < 3 (cannot test clustering).
    
    Notes:
    ------
    - Silhouette scores range from -1 to 1, where higher is better
    - Returns the k with highest average silhouette score
    - If all groups end up in one cluster, that solution is not selected
    - max_k is computed as n_samples - 1 (one less than the number of groups)
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import numpy as np
    
    n_samples = features.shape[0]
    max_k = n_samples - 1
    
    if max_k < 2:
        return 1
    
    best_k = 2
    best_silhouette = -1
    
    for k in range(2, max_k + 1):
        # Suppress sklearn convergence warnings during optimal k search
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Number of distinct clusters.*found smaller than n_clusters.*")
            # Use replication-specific seed for silhouette selection
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            
            # Calculate silhouette score only if we have more than 1 cluster
            actual_clusters = len(np.unique(cluster_labels))
            if actual_clusters > 1:
                silhouette_avg = silhouette_score(features, cluster_labels)
                if silhouette_avg > best_silhouette:
                    best_silhouette = silhouette_avg
                    best_k = k
    
    return best_k


def visualize_kmeans_network_plot(scenario_result, item_idx, save_plots=False, results_folder=None,
                                  r=None, n=None, p=None, groups_name=None, show_plots=True,
                                  legend_fontsize=12):
    """
    Create a 2D network plot showing K-means clusters as centroids connected to their groups.
    
    This function visualizes K-means clustering results as a network diagram where cluster
    centroids (squares) are connected to the groups (circles) assigned to them. Groups in
    the same cluster share the same color.
    
    Parameters:
    -----------
    scenario_result : dict
        Results for a specific scenario from K-means clustering, containing 'item_results'
        with clustering information for each item.
    item_idx : int
        Zero-based item index to visualize.
    save_plots : bool, optional
        Whether to save the plot to a file.
        Default: False
    results_folder : str, optional
        Directory path to save plots. If None, uses 'kmeans_plots'.
        Default: None
    r : int, optional
        Replication number for filename generation.
    n : int, optional
        Sample size for filename generation.
    p : int, optional
        DIF percentage for filename generation.
    groups_name : str, optional
        Groups string (e.g., 'Ten') for filename generation.
    show_plots : bool, optional
        Whether to display the plot interactively.
        Default: True
    legend_fontsize : int, optional
        Font size for the legend.
        Default: 12
        
    Returns:
    --------
    None
        Plot is displayed and/or saved to file.
    
    Notes:
    ------
    - Plot filename pattern: 'K_Means_Clustering_{groups_name}Groups_{dif_type}_N{n}_P{p}_R{r}_Item{item}.png'
    - Cluster centroids are displayed as squares in inner circle
    - Groups are displayed as circles in outer circle
    - Lines connect each group to its assigned cluster centroid
    - All edges use consistent line width and alpha for uniformity
    """
    # Disable interactive mode if not showing plots to prevent IDE from displaying
    if not show_plots:
        plt.ioff()
    
    if item_idx not in scenario_result['item_results']:
        print(f"Item {item_idx} not found in results")
        return
    
    item_result = scenario_result['item_results'][item_idx]
    cluster_dict = item_result['final_clustering']['cluster_dict']
    kmeans_data = item_result['kmeans_specific']
    groups = item_result['groups']
    dif_type = item_result['dif_type']
    dif_matrix = item_result['dif_matrix']
    
    # Get cluster information
    n_clusters = kmeans_data['n_clusters']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Define colors for clusters
    colors = plt.cm.Set2(np.linspace(0, 1, n_clusters))
    
    # Calculate positions for groups in a circle
    n_groups = len(groups)
    angles = np.linspace(0, 2*np.pi, n_groups, endpoint=False)
    group_positions = {}
    
    # Place groups on outer circle
    radius_groups = 3.0
    for i, group in enumerate(groups):
        x = radius_groups * np.cos(angles[i])
        y = radius_groups * np.sin(angles[i])
        group_positions[group] = (x, y)
    
    # Calculate positions for cluster centroids
    unique_clusters = sorted(set(cluster_dict.values()))
    centroid_positions = {}
    
    if len(unique_clusters) == 1:
        # If only one cluster, place centroid at origin
        centroid_positions[unique_clusters[0]] = (0, 0)
    else:
        # Place centroids in inner circle
        radius_centroids = 1.0
        centroid_angles = np.linspace(0, 2*np.pi, len(unique_clusters), endpoint=False)
        for i, cluster_id in enumerate(unique_clusters):
            x = radius_centroids * np.cos(centroid_angles[i])
            y = radius_centroids * np.sin(centroid_angles[i])
            centroid_positions[cluster_id] = (x, y)
    
    # Draw connections from centroids to groups
    for group, cluster_id in cluster_dict.items():
        group_pos = group_positions[group]
        centroid_pos = centroid_positions[cluster_id]
        
        # Use constant line properties for complete uniformity
        line_width = 4  # Fixed line width
        alpha = 1.0     # Fixed alpha for all lines (darkened from 0.8)
        
        ax.plot([centroid_pos[0], group_pos[0]], [centroid_pos[1], group_pos[1]], 
                color=colors[cluster_id-1], linewidth=line_width, alpha=alpha, zorder=1)
    
    # Draw cluster centroids
    for cluster_id, pos in centroid_positions.items():
        ax.scatter(pos[0], pos[1], s=800, c=[colors[cluster_id-1]], 
                  marker='s', edgecolors='black', linewidths=2, zorder=3, alpha=0.9)
        ax.text(pos[0], pos[1], f'C{cluster_id}', ha='center', va='center', 
                fontsize=12, fontweight='bold', color='black', zorder=4)
    
    # Draw groups
    for i, (group, pos) in enumerate(group_positions.items()):
        cluster_id = cluster_dict[group]
        ax.scatter(pos[0], pos[1], s=400, c=[colors[cluster_id-1]], 
                  marker='o', edgecolors='black', linewidths=1.5, zorder=2, alpha=0.8)
        
        # Add group labels
        label_offset = 0.3
        label_x = pos[0] + label_offset * np.cos(angles[i])
        label_y = pos[1] + label_offset * np.sin(angles[i])
        ax.text(label_x, label_y, group.replace('Group', 'G'), 
                ha='center', va='center', fontsize=10, fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
                zorder=5)
    
    # Create legend
    legend_elements = []
    for i, cluster_id in enumerate(unique_clusters):
        cluster_groups = [g for g, c in cluster_dict.items() if c == cluster_id]
        label = f'Cluster {cluster_id} ({len(cluster_groups)} groups)'
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=colors[i], markersize=10,
                                        label=label, markeredgecolor='black'))
    
    # Place legend at bottom center, outside the plot area
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), 
              ncol=min(len(unique_clusters), 5), fontsize=legend_fontsize)
    
    # Set plot properties
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
   
    # Remove axis ticks and labels for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plots:
        plot_dir = results_folder if results_folder else "kmeans_plots"
        os.makedirs(plot_dir, exist_ok=True)
        
        # Create sensible filename with scenario information
        if r is not None and n is not None and p is not None and groups_name is not None:
            filename = f"{plot_dir}/K_Means_Clustering_{groups_name}Groups_{dif_type}_N{n}_P{p}_R{r}_Item{item_idx+1}.png"
        else:
            # Fallback to simpler name if scenario info not provided
            filename = f"{plot_dir}/K_Means_Clustering_{dif_type}_Item{item_idx+1}.png"
            
        plt.savefig(filename, dpi=720, bbox_inches='tight')
        print(f"Plot saved to: {filename}")
    
    if show_plots:
        plt.show()
    
    # Always close the figure to prevent memory issues
    plt.close()
    
    # Restore interactive mode
    plt.ion()


def determine_optimal_clusters_hierarchical(features, linkage_matrix, verbose=False):
    """
    Determine optimal number of clusters for hierarchical clustering using silhouette score.
    
    This function uses only the silhouette method with ward linkage and euclidean distance
    to find the optimal number of clusters.
    
    Parameters:
    -----------
    features : np.array
        Feature matrix used for clustering (n_samples x n_features).
    linkage_matrix : np.array
        Linkage matrix from hierarchical clustering.
    verbose : bool, optional
        Whether to print details.
        Default: False
        
    Returns:
    --------
    int
        Optimal number of clusters based on silhouette score.
        Returns 2 if n_samples < 3 (minimum meaningful clustering).
    
    Notes:
    ------
    - Silhouette scores range from -1 to 1, where higher is better
    - Returns the k with highest average silhouette score
    - If clustering produces only 1 cluster, that solution gets score of -1
    - max_clusters is computed as n_samples - 1 (one less than the number of groups)
    """
    n_samples = features.shape[0]
    max_clusters = n_samples - 1
    
    if max_clusters <= 1:
        return 2  # Minimum meaningful clustering
    
    # Use silhouette analysis to find optimal number of clusters
    silhouette_scores = []
    k_range = range(2, max_clusters + 1)
    
    for k in k_range:
        cluster_labels = fcluster(linkage_matrix, k, criterion='maxclust')
        if len(np.unique(cluster_labels)) > 1:  # Need at least 2 clusters for silhouette
            score = silhouette_score(features, cluster_labels)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(-1)  # Invalid clustering gets low score
    
    if silhouette_scores:
        best_k_index = np.argmax(silhouette_scores)
        optimal_k = k_range[best_k_index]
        best_silhouette = silhouette_scores[best_k_index]
        
        if verbose:
            print(f"Best silhouette score: {best_silhouette:.3f} at K={optimal_k}")
    else:
        optimal_k = 2  # If no valid silhouette scores, default to minimum K=2
        
    if verbose:
        print(f"Silhouette scores: {dict(zip(k_range, silhouette_scores))}")
        print(f"Optimal k (silhouette): {optimal_k}")
        
    return optimal_k


def apply_hierarchical_to_dif_matrix(dif_matrix, groups, dif_type, item_index, 
                                     n_clusters=None, verbose=False):
    """
    Apply hierarchical clustering to a single DIF matrix using ward linkage and euclidean distance.
    
    This function extracts features from a DIF probability matrix and applies hierarchical
    clustering to identify groups with similar DIF patterns. The optimal number of clusters
    is determined automatically using the silhouette score if not specified.
    
    Parameters:
    -----------
    dif_matrix : pd.DataFrame
        DIF probability matrix for a single item, with groups as both rows and columns.
        Values represent pairwise DIF probabilities between groups.
    groups : list of str
        List of group names (e.g., ['Group1', 'Group2', 'Group3']).
    dif_type : str
        Type of DIF being analyzed ('DIF_a' for discrimination or 'DIF_b' for difficulty).
    item_index : int
        Zero-based item index being analyzed.
    n_clusters : int or None, optional
        Number of clusters. If None, will auto-determine using silhouette method.
        Default: None
    verbose : bool, optional
        Whether to print detailed clustering information.
        Default: False
        
    Returns:
    --------
    dict
        Clustering results containing:
        - 'cluster_dict': dict mapping group names to cluster IDs (1-indexed)
        - 'connected_components': list of sets, each set contains groups in a cluster
        - 'edges': list of tuples (group1, group2, dif_value) for groups in same cluster
        - 'features': array of original DIF features
        - 'features_scaled': array of standardized features
        - 'linkage_matrix': hierarchical clustering linkage matrix
        - 'distances': pairwise distance matrix
        - 'silhouette_score': silhouette score of the clustering (-1 to 1)
        - 'inertia': within-cluster sum of squares
        - 'n_clusters': actual number of clusters found
        - 'requested_clusters': number of clusters requested
        - 'linkage_method': 'ward'
        - 'distance_metric': 'euclidean'
    
    Notes:
    ------
    - Uses StandardScaler to normalize features before clustering
    - Uses ward linkage (minimizes within-cluster variance)
    - Uses euclidean distance (required for ward linkage)
    - Handles NaN values by replacing with 0.5 (neutral DIF probability)
    - Silhouette score of 0 indicates single cluster, -1 indicates degenerate cases
    """
    from sklearn.preprocessing import StandardScaler
    
    # Extract features from DIF matrix
    n_groups = len(groups)
    features = []
    
    for i, group in enumerate(groups):
        # For each group, create a feature vector of its DIF values with other groups
        group_features = []
        for j, other_group in enumerate(groups):
            if i != j:  # Exclude self-comparison
                dif_value = dif_matrix.loc[group, other_group]
                # Handle NaN values
                if pd.isna(dif_value):
                    dif_value = 0.5  # Neutral value for missing DIF
                group_features.append(dif_value)
        features.append(group_features)
    
    features = np.array(features)
    
    # Standardize features (important for distance-based clustering)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Compute distance matrix using euclidean distance (required for ward linkage)
    distances = pdist(features_scaled, metric='euclidean')
    
    # Perform hierarchical clustering using ward linkage
    linkage_matrix = linkage(distances, method='ward')
    
    # Determine optimal number of clusters if not provided
    if n_clusters is None:
        n_clusters = determine_optimal_clusters_hierarchical(
            features_scaled, linkage_matrix, verbose=verbose
        )
    
    # Get cluster assignments
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    # Check actual number of clusters found
    actual_n_clusters = len(np.unique(cluster_labels))
    
    # Create cluster dictionary (same format as K-means method)
    cluster_dict = {}
    for i, group in enumerate(groups):
        cluster_dict[group] = cluster_labels[i]  # Already 1-indexed from fcluster
    
    # Calculate clustering quality metrics
    if actual_n_clusters > 1 and actual_n_clusters < n_groups:
        silhouette_avg = silhouette_score(features_scaled, cluster_labels)
        
        # Calculate inertia equivalent (within-cluster sum of squares)
        inertia = 0
        for cluster_id in np.unique(cluster_labels):
            cluster_points = features_scaled[cluster_labels == cluster_id]
            if len(cluster_points) > 1:
                cluster_center = np.mean(cluster_points, axis=0)
                inertia += np.sum((cluster_points - cluster_center) ** 2)
    else:
        silhouette_avg = 0 if actual_n_clusters == 1 else -1  # -1 for degenerate cases
        inertia = 0
    
    # Create "edges" for compatibility with visualization
    # Groups in same cluster are considered "connected"
    edges = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if cluster_labels[i] == cluster_labels[j]:
                dif_value = dif_matrix.loc[groups[i], groups[j]]
                if pd.isna(dif_value):
                    dif_value = 0.5
                edges.append((groups[i], groups[j], dif_value))
    
    # Group clusters for connected components format
    connected_components = []
    unique_clusters = np.unique(cluster_labels)
    for cluster_id in unique_clusters:
        component = [groups[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
        connected_components.append(set(component))
    
    if verbose:
        print(f"Hierarchical Clustering Results for {dif_type} - Item {item_index + 1}")
        print(f"Requested clusters: {n_clusters}, Actual clusters found: {actual_n_clusters}")
        print(f"Linkage method: ward")
        print(f"Distance metric: euclidean")
        print(f"Silhouette score: {silhouette_avg:.3f}")
        print(f"Inertia: {inertia:.3f}")
        if actual_n_clusters == 1:
            print("Note: All groups clustered together (no significant DIF detected)")
        print("Cluster assignments:")
        for group, cluster_id in cluster_dict.items():
            print(f"  {group}: Cluster {cluster_id}")
    
    return {
        'cluster_dict': cluster_dict,
        'connected_components': connected_components,
        'edges': edges,
        'features': features,
        'features_scaled': features_scaled,
        'linkage_matrix': linkage_matrix,
        'distances': distances,
        'silhouette_score': silhouette_avg,
        'inertia': inertia,
        'n_clusters': actual_n_clusters,  # Use actual clusters found
        'requested_clusters': n_clusters,  # Keep track of what was requested
        'linkage_method': 'ward',
        'distance_metric': 'euclidean'
    }


def visualize_hierarchical_network_plot(scenario_result, item_idx, save_plots=False, results_folder=None,
                                        r=None, n=None, p=None, groups_name=None, show_plots=True,
                                        legend_fontsize=12):
    """
    Create a dendrogram visualization showing hierarchical clustering of groups.
    
    This function visualizes hierarchical clustering results as a dendrogram, showing
    the tree structure of how groups are merged based on their DIF similarity. Clusters
    are color-coded to show the final cluster assignments.
    
    Parameters:
    -----------
    scenario_result : dict
        Results for a specific scenario from hierarchical clustering, containing 'item_results'
        with clustering information for each item.
    item_idx : int
        Zero-based item index to visualize.
    save_plots : bool, optional
        Whether to save the plot to a file.
        Default: False
    results_folder : str, optional
        Directory path to save plots. If None, uses 'hierarchical_plots'.
        Default: None
    r : int, optional
        Replication number for filename generation.
    n : int, optional
        Sample size for filename generation.
    p : int, optional
        DIF percentage for filename generation.
    groups_name : str, optional
        Groups string (e.g., 'Ten') for filename generation.
    show_plots : bool, optional
        Whether to display the plot interactively.
        Default: True
    legend_fontsize : int, optional
        Font size for the legend.
        Default: 12
        
    Returns:
    --------
    None
        Plot is displayed and/or saved to file.
    
    Notes:
    ------
    - Plot filename pattern: 'Hierarchical_Dendrogram_{groups_name}Groups_{dif_type}_N{n}_P{p}_R{r}_Item{item}.png'
    - Uses scipy's dendrogram function for proper tree visualization
    - Clusters are color-coded based on the optimal number of clusters determined by silhouette score
    - Group labels are displayed at the bottom of the dendrogram
    """
    from scipy.cluster.hierarchy import dendrogram, fcluster
    
    # Disable interactive mode if not showing plots to prevent IDE from displaying
    if not show_plots:
        plt.ioff()
    
    if item_idx not in scenario_result['item_results']:
        print(f"Item {item_idx} not found in results")
        return
    
    item_result = scenario_result['item_results'][item_idx]
    cluster_dict = item_result['final_clustering']['cluster_dict']
    hierarchical_data = item_result['hierarchical_specific']
    groups = item_result['groups']
    dif_type = item_result['dif_type']
    
    # Get hierarchical clustering data
    linkage_matrix = hierarchical_data['linkage_matrix']
    n_clusters = hierarchical_data['n_clusters']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create short labels for groups (G1, G2, etc.)
    short_labels = [g.replace('Group', 'G') for g in groups]
    
    # Calculate color threshold for dendrogram coloring
    # This determines at what height the dendrogram branches get different colors
    if n_clusters > 1:
        # Find the appropriate threshold to get the desired number of clusters
        # The threshold should be just below the merge height that would reduce clusters
        max_d = linkage_matrix[-n_clusters + 1, 2] if n_clusters <= len(linkage_matrix) else 0
        color_threshold = max_d * 0.99  # Slightly below to ensure proper coloring
    else:
        color_threshold = 0
    
    # Create dendrogram with colored clusters
    dendro = dendrogram(
        linkage_matrix,
        labels=short_labels,
        ax=ax,
        leaf_rotation=45,
        leaf_font_size=10,
        color_threshold=color_threshold,
        above_threshold_color='gray'
    )
    
    # Customize appearance
    ax.set_ylabel('Distance', fontsize=12)
    ax.set_xlabel('Groups', fontsize=12)
    
    # Add title showing clustering info
    silhouette = hierarchical_data.get('silhouette_score', None)
    if silhouette is not None:
        title = f'Hierarchical Clustering Dendrogram - {dif_type} Item {item_idx + 1}\n'
        title += f'Optimal Clusters: {n_clusters}, Silhouette Score: {silhouette:.3f}'
    else:
        title = f'Hierarchical Clustering Dendrogram - {dif_type} Item {item_idx + 1}\n'
        title += f'Optimal Clusters: {n_clusters}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add horizontal line showing where clusters are cut
    if n_clusters > 1 and n_clusters <= len(linkage_matrix):
        cut_height = linkage_matrix[-n_clusters + 1, 2]
        ax.axhline(y=cut_height, color='red', linestyle='--', linewidth=2, 
                   label=f'Cut for {n_clusters} clusters')
        # Place legend at bottom center, outside the plot area
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=legend_fontsize)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if requested
    if save_plots:
        plot_dir = results_folder if results_folder else "hierarchical_plots"
        os.makedirs(plot_dir, exist_ok=True)
        
        # Create sensible filename with scenario information
        if r is not None and n is not None and p is not None and groups_name is not None:
            filename = f"{plot_dir}/Hierarchical_Dendrogram_{groups_name}Groups_{dif_type}_N{n}_P{p}_R{r}_Item{item_idx+1}.png"
        else:
            # Fallback to simpler name if scenario info not provided
            filename = f"{plot_dir}/Hierarchical_Dendrogram_{dif_type}_Item{item_idx+1}.png"
            
        plt.savefig(filename, dpi=720, bbox_inches='tight')
        print(f"Plot saved to: {filename}")
    
    if show_plots:
        plt.show()
    
    # Always close the figure to prevent memory issues
    plt.close()
    
    # Restore interactive mode
    plt.ion()


def visualize_connected_components_per_item_tdc(G, cluster_dict,
                                          dif_type='DIF_a', item_index=0, save_plot=False, 
                                          base_filename=None,
                                          original_edges=None, transitive_edges=None,
                                          show_title=True, results_folder=None):
    """
    Visualize connected components for TDC single-file analysis with base_filename naming.
    
    This is a specialized version of visualize_connected_components_per_item for the TDC() function
    that uses base_filename for file naming instead of n/p/r pattern.
    """
    
    if G is None or G.number_of_nodes() == 0:
        print(f"No graph data available for {dif_type} Item {item_index + 1}")
        return
    
    # Set up the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create layout
    pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
    
    # Get unique clusters and assign colors
    unique_clusters = sorted(set(cluster_dict.values()))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
    cluster_colors = {cluster: colors[i] for i, cluster in enumerate(unique_clusters)}
    
    # Color nodes by cluster
    node_colors = [cluster_colors[cluster_dict[node]] for node in G.nodes()]
    
    # Draw original edges (thin, black)
    if original_edges:
        nx.draw_networkx_edges(G, pos, edgelist=original_edges, 
                              edge_color='black', width=1, alpha=1.0, ax=ax)
    
    # Draw transitive edges (thick, red)  
    if transitive_edges:
        nx.draw_networkx_edges(G, pos, edgelist=transitive_edges,
                              edge_color='red', width=3, alpha=0.8, ax=ax)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=800, alpha=0.9, ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
    
    # Create legend for edge types only (no cluster names)
    legend_elements = []
    
    # Add edge type legend if there are transitive edges
    if transitive_edges:
        legend_elements.extend([
            plt.Line2D([0], [0], color='black', lw=2, alpha=1.0, label='Original Edges'),
            plt.Line2D([0], [0], color='red', lw=3, alpha=0.8, label='Transitive Edges')
        ])
    
    # Only show legend if there are legend elements (i.e., if there are transitive edges)
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)
    
    # Set title
    if show_title:
        if dif_type == 'DIF_a':
            title = f'TDC Results\nDIF on a'
        else:  # DIF_b
            title = f'TDC Results\nDIF on b'
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.set_aspect('equal')
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        if results_folder and base_filename:
            filename = f"DIF_Clustering_Components_{base_filename}_{dif_type}_Item{item_index + 1}.png"
            filepath = os.path.join(results_folder, filename)
            plt.savefig(filepath, dpi=720, bbox_inches='tight')
            print(f"  Plot saved: {filename}")
    
    plt.show()


def visualize_connected_components_side_by_side_tdc(detailed_results, item_idx, base_filename,
                                                   save_plot=False, show_title=True, results_folder=None):
    """
    Create side-by-side visualization of DIF_a and DIF_b clustering results for a single item.
    
    Parameters:
    -----------
    detailed_results : dict
        Dictionary containing results for both 'DIF_a' and 'DIF_b' from DIF_Cluster_Components_Per_Item
    item_idx : int
        Index of the item to visualize
    base_filename : str
        Base filename for saving plots
    save_plot : bool, optional
        Whether to save the plot (default: False)
    show_title : bool, optional
        Whether to show plot titles (default: True)
    results_folder : str, optional
        Folder to save plots in (default: None)
    """
    
    # Check if we have results for both DIF types and the specific item
    dif_a_available = ('DIF_a' in detailed_results and 
                       detailed_results['DIF_a'] is not None and 
                       item_idx in detailed_results['DIF_a']['item_results'])
    
    dif_b_available = ('DIF_b' in detailed_results and 
                       detailed_results['DIF_b'] is not None and 
                       item_idx in detailed_results['DIF_b']['item_results'])
    
    if not (dif_a_available or dif_b_available):
        print(f"No clustering data available for Item {item_idx + 1}")
        return
    
    # Set up the figure with subplots - reduce spacing between plots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    plt.subplots_adjust(wspace=0.1)  # Reduce horizontal spacing between subplots
    
    # Helper function to plot a single DIF type
    def plot_single_dif(ax, dif_type, item_res, title_suffix):
        if item_res is None:
            ax.text(0.5, 0.5, f'No {dif_type} data available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            return
        
        # Extract components for plotting
        G = item_res['final_clustering']['graph']
        cluster_dict = item_res['final_clustering']['cluster_dict']
        groups = item_res['groups']
        threshold = item_res['recommended_threshold']
        original_edges = item_res['final_clustering']['original_edges']
        transitive_edges = item_res['final_clustering']['transitive_edges']
        
        if G is None or G.number_of_nodes() == 0:
            ax.text(0.5, 0.5, f'No {dif_type} connections found', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            return
        
        # Create layout (use same seed for consistency across subplots)
        pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
        
        # Get unique clusters and assign colors
        unique_clusters = sorted(set(cluster_dict.values()))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
        cluster_colors = {cluster: colors[i] for i, cluster in enumerate(unique_clusters)}
        
        # Color nodes by cluster
        node_colors = [cluster_colors[cluster_dict[node]] for node in G.nodes()]
        
        # Draw original edges (thin, black)
        if original_edges:
            nx.draw_networkx_edges(G, pos, edgelist=original_edges, 
                                  edge_color='black', width=1, alpha=1.0, ax=ax)
        
        # Draw transitive edges (thick, red)  
        if transitive_edges:
            nx.draw_networkx_edges(G, pos, edgelist=transitive_edges,
                                  edge_color='red', width=3, alpha=0.8, ax=ax)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=800, alpha=0.9, ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
        
        # Create legend for edge types only (no cluster names)
        legend_elements = []
        
        # Add edge type legend if there are transitive edges
        if transitive_edges:
            legend_elements.extend([
                plt.Line2D([0], [0], color='black', lw=2, alpha=1.0, label='Original Edges'),
                plt.Line2D([0], [0], color='red', lw=3, alpha=0.8, label='Transitive Edges')
            ])
        
        # Only show legend if there are legend elements (i.e., if there are transitive edges)
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)
        
        # Set title
        if show_title:
            if dif_type == 'DIF_a':
                title = f'TDC Plot for DIF on a - Item {item_idx + 1}'
            else:  # DIF_b
                title = f'TDC Plot for DIF on b - Item {item_idx + 1}'
            ax.set_title(title, fontsize=12, fontweight='bold')
        
        ax.set_aspect('equal')
    
    # Plot DIF_a on left subplot
    if dif_a_available:
        item_res_a = detailed_results['DIF_a']['item_results'][item_idx]
        plot_single_dif(axes[0], 'DIF_a', item_res_a, ' (Discrimination)')
    else:
        plot_single_dif(axes[0], 'DIF_a', None, ' (Discrimination)')
    
    # Plot DIF_b on right subplot
    if dif_b_available:
        item_res_b = detailed_results['DIF_b']['item_results'][item_idx]
        plot_single_dif(axes[1], 'DIF_b', item_res_b, ' (Difficulty)')
    else:
        plot_single_dif(axes[1], 'DIF_b', None, ' (Difficulty)')
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot and results_folder and base_filename:
        filename = f"DIF_Clustering_Components_{base_filename}_Combined_Item{item_idx + 1}.png"
        filepath = os.path.join(results_folder, filename)
        plt.savefig(filepath, dpi=720, bbox_inches='tight')
        print(f"  Combined plot saved: {filename}")
    
    plt.show()


def export_stan_constraints_csv_tdc(stan_constraints, base_filename, verbose=False, results_folder=None):
    """
    Export Stan constraint matrices to CSV files for TDC single-file analysis.
    
    This is a specialized version of export_stan_constraints_csv for the TDC() function
    that uses base_filename for file naming instead of groups/n/p/r pattern.
    
    Parameters:
    -----------
    stan_constraints : dict
        Constraint matrices and metadata from create_stan_constraints_from_clustering()
    base_filename : str
        Base name for output files
    verbose : bool, optional
        Whether to print verbose output (default: False)
    results_folder : str, optional
        Folder where files will be saved (default: None, saves to current directory)
    """
    # Create DataFrames with group names as columns
    group_names = stan_constraints['groups']
    
    # Discrimination constraints (a parameters)
    a_df = pd.DataFrame(
        stan_constraints['a_constraint_group'], 
        columns=[f"Group{i+1}" for i in range(len(group_names))],
        index=[f"Item{i+1}" for i in range(stan_constraints['n_items'])]
    )
    
    # Difficulty constraints (b parameters)
    b_df = pd.DataFrame(
        stan_constraints['b_constraint_group'], 
        columns=[f"Group{i+1}" for i in range(len(group_names))],
        index=[f"Item{i+1}" for i in range(stan_constraints['n_items'])]
    )
    
    # Save CSV files with base_filename naming convention
    a_filename = f"Estimated_Stan_a_constraints_{base_filename}.csv"
    b_filename = f"Estimated_Stan_b_constraints_{base_filename}.csv"
    
    # Construct full file paths with optional folder
    if results_folder:
        a_filepath = os.path.join(results_folder, a_filename)
        b_filepath = os.path.join(results_folder, b_filename)
    else:
        a_filepath = a_filename
        b_filepath = b_filename
    
    # Save constraint matrices
    a_df.to_csv(a_filepath, index=True)
    b_df.to_csv(b_filepath, index=True)
    
    # Create and save metadata file
    metadata = {
        'base_filename': base_filename,
        'n_items': stan_constraints['n_items'],
        'n_groups': stan_constraints['n_groups'],
        'N_a_constraints': stan_constraints['N_a_constraints'],
        'N_b_constraints': stan_constraints['N_b_constraints'],
        'group_names': group_names,
        'files_created': [a_filename, b_filename]
    }
    
    metadata_filename = f"Estimated_Stan_metadata_{base_filename}.json"
    
    # Construct full file path with optional folder
    if results_folder:
        metadata_filepath = os.path.join(results_folder, metadata_filename)
    else:
        metadata_filepath = metadata_filename
    
    with open(metadata_filepath, 'w') as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"Stan constraint files saved:")
        print(f"  {a_filename}")
        print(f"  {b_filename}")
        print(f"  {metadata_filename}")


def export_stan_constraints_csv(stan_constraints, groups, n, p, r, verbose = False, results_folder=None):
    """
    Export Stan constraint matrices to CSV files.
    
    Parameters:
    -----------
    stan_constraints : dict
        Constraint matrices and metadata from create_stan_constraints_from_clustering()
    groups : str
        Group identifier (e.g., "Ten", "Three")
    n : int
        Sample size
    p : int
        DIF percentage
    r : int
        Replication number
    verbose : bool, optional
        Whether to print verbose output (default: False)
    results_folder : str, optional
        Folder where files will be saved (default: None, saves to current directory)
    """
    # Create DataFrames with group names as columns
    group_names = stan_constraints['groups']
    
    # Discrimination constraints (a parameters)
    a_df = pd.DataFrame(
        stan_constraints['a_constraint_group'], 
        columns=[f"Group{i+1}" for i in range(len(group_names))],
        index=[f"Item{i+1}" for i in range(stan_constraints['n_items'])]
    )
    
    # Difficulty constraints (b parameters)
    b_df = pd.DataFrame(
        stan_constraints['b_constraint_group'], 
        columns=[f"Group{i+1}" for i in range(len(group_names))],
        index=[f"Item{i+1}" for i in range(stan_constraints['n_items'])]
    )
    
    # Save CSV files with specified naming convention (matching MIRT pattern)
    a_filename = f"Estimated_Stan_a_constraints_TDC_{groups}_{n}_{p}_Replication{r}.csv"
    b_filename = f"Estimated_Stan_b_constraints_TDC_{groups}_{n}_{p}_Replication{r}.csv"
    
    # Construct full file paths with optional folder
    if results_folder:
        a_filepath = os.path.join(results_folder, a_filename)
        b_filepath = os.path.join(results_folder, b_filename)
    else:
        a_filepath = a_filename
        b_filepath = b_filename
    
    # Save constraint matrices
    a_df.to_csv(a_filepath, index=True)
    b_df.to_csv(b_filepath, index=True)
    
    # Create and save metadata file
    metadata = {
        'groups': groups,
        'n_sample_size': n,
        'p_dif_percentage': p,
        'r_replication': r,
        'n_items': stan_constraints['n_items'],
        'n_groups': stan_constraints['n_groups'],
        'N_a_constraints': stan_constraints['N_a_constraints'],
        'N_b_constraints': stan_constraints['N_b_constraints'],
        'group_names': group_names,
        'files_created': [a_filename, b_filename]
    }
    
    metadata_filename = f"Estimated_Stan_metadata_TDC_{groups}_{n}_{p}_Replication{r}.json"
    
    # Construct full file path with optional folder
    if results_folder:
        metadata_filepath = os.path.join(results_folder, metadata_filename)
    else:
        metadata_filepath = metadata_filename
    
    import json
    with open(metadata_filepath, 'w') as f:
        json.dump(metadata, f, indent=2)


# Main Function Calls

def DIF_Detection(data_filename, model_folder_path, 
                  labels_filename=None, verbose=False, save_results=False,
                  output_filename=None, calculate_performance=False, results_folder=None):
    """
    Apply trained DIF detection models to user-specified dataset by loading models from saved folder.
    
    This function loads a previously trained InterDIFNet model from a specified folder and applies
    it to new data for DIF detection. The function enforces strict data quality requirements - it will
    stop and raise an error if any required features are missing or contain missing values (NaN/null).
    It can optionally calculate performance metrics if true labels are provided.

    Parameters:
    -----------
    data_filename : str
        Path to the empirical data CSV file containing features
    model_folder_path : str
        Path to the folder containing saved model files (created by train_InterDIFNet)
        Expected files in folder:
        - merged_model_*groups.keras OR model_dif_a_*groups.keras & model_dif_b_*groups.keras
        - scaler_*groups.pkl
        - optimal_thresholds_*groups.json
        - model_metadata_*groups.json
        - feature_names_*groups.txt
    labels_filename : str, optional
        Path to CSV file with true DIF labels for performance evaluation (default: None)
    verbose : bool
        Controls detailed printing (default: False)
    save_results : bool
        Whether to save probability and prediction results to CSV files (default: False)
    output_filename : str, optional
        Custom base filename for saved results. If None, auto-generates from data_filename
    calculate_performance : bool
        Whether to calculate performance metrics (requires labels_filename) (default: False)
    results_folder : str, optional
        Folder path where results will be saved. If None, defaults to "{groups}_Group_InterDIFNet_Results"
        where {groups} is extracted from the model metadata. Folder will be created if it doesn't exist.

    Returns:
    --------
    dict
        Dictionary containing:
        - 'probabilities_a': DataFrame with DIF_a probabilities for each item
        - 'probabilities_b': DataFrame with DIF_b probabilities for each item
        - 'predictions': DataFrame with binary predictions (thresholded)
        - 'all_probabilities': DataFrame with all probabilities combined
        - 'performance_metrics': DataFrame with TP/FP/TN/FN counts and TPR/FPR (if calculate_performance=True)
        - 'model_info': Dictionary with loaded model metadata

    Examples:
    --------
    # Basic DIF detection without performance evaluation
    >>> results = DIF_Detection(
    ...     data_filename="my_test_data.csv",
    ...     model_folder_path="Trained_Three_Group_InterDIFNet"
    ... )
    >>> print("DIF probabilities:", results['all_probabilities'])
    
    # DIF detection with performance evaluation and custom results folder
    >>> results = DIF_Detection(
    ...     data_filename="test_features.csv",
    ...     model_folder_path="My_Custom_Model_Folder",
    ...     labels_filename="test_labels.csv",
    ...     calculate_performance=True,
    ...     save_results=True,
    ...     results_folder="My_DIF_Results",
    ...     verbose=True
    ... )
    >>> print("Performance metrics:", results['performance_metrics'])
    
    # Using saved results for further analysis
    >>> tpr_dif_a = results['performance_metrics'].loc[
    ...     results['performance_metrics']['Group'] == 'DIF_a', 'TPR'
    ... ].values[0]

    Notes:
    ------
    - Data file MUST contain ALL features used during model training (no missing features allowed)
    - Data file MUST NOT contain any missing values (NaN/null) in required feature columns
    - Features will be automatically reordered to match training model expectations  
    - Function will raise ValueError if any required features are missing or contain missing data
    - If calculate_performance=True, labels file must have same column structure as training labels
    - Model metadata is automatically loaded to ensure compatibility
    - Extra features in data (not used in training) will be ignored
    """
    
    # Load model metadata to get model configuration
    metadata_files = [f for f in os.listdir(model_folder_path) if f.startswith('model_metadata_') and f.endswith('.json')]
    if not metadata_files:
        raise FileNotFoundError(f"No model metadata file found in {model_folder_path}")
    
    metadata_path = os.path.join(model_folder_path, metadata_files[0])
    with open(metadata_path, 'r') as f:
        model_metadata = json.load(f)
    
    merged = model_metadata['model_info']['merged']
    groups = model_metadata['model_info']['groups']
    set1_cols = model_metadata['label_info']['set1_cols']
    set2_cols = model_metadata['label_info']['set2_cols']
    feature_names = model_metadata['feature_info']['feature_names']
    
    if verbose:
        print(f"Loading {model_metadata['model_info']['model_type']} model for {groups} groups")
        print(f"Model was trained with {len(feature_names)} features")
    
    # Load models
    if merged:
        model_files = [f for f in os.listdir(model_folder_path) if f.startswith('merged_model_') and f.endswith('.keras')]
        if not model_files:
            raise FileNotFoundError(f"No merged model file found in {model_folder_path}")
        model_path = os.path.join(model_folder_path, model_files[0])
        model_dif_a = tf.keras.models.load_model(model_path)
        model_dif_b = None
        if verbose:
            print(f"Loaded merged model from: {model_files[0]}")
    else:
        # Load separate models
        model_a_files = [f for f in os.listdir(model_folder_path) if f.startswith('model_dif_a_') and f.endswith('.keras')]
        model_b_files = [f for f in os.listdir(model_folder_path) if f.startswith('model_dif_b_') and f.endswith('.keras')]
        
        if not model_a_files or not model_b_files:
            raise FileNotFoundError(f"Missing model files in {model_folder_path}")
        
        model_a_path = os.path.join(model_folder_path, model_a_files[0])
        model_b_path = os.path.join(model_folder_path, model_b_files[0])
        
        model_dif_a = tf.keras.models.load_model(model_a_path)
        model_dif_b = tf.keras.models.load_model(model_b_path)
        
        if verbose:
            print(f"Loaded separate models from: {model_a_files[0]}, {model_b_files[0]}")
    
    # Load scaler
    scaler_files = [f for f in os.listdir(model_folder_path) if f.startswith('scaler_') and f.endswith('.pkl')]
    if not scaler_files:
        raise FileNotFoundError(f"No scaler file found in {model_folder_path}")
    
    scaler_path = os.path.join(model_folder_path, scaler_files[0])
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load optimal thresholds
    threshold_files = [f for f in os.listdir(model_folder_path) if f.startswith('optimal_thresholds_') and f.endswith('.json')]
    if not threshold_files:
        raise FileNotFoundError(f"No threshold file found in {model_folder_path}")
    
    threshold_path = os.path.join(model_folder_path, threshold_files[0])
    with open(threshold_path, 'r') as f:
        threshold_data = json.load(f)
    
    opt_thr_a = threshold_data['opt_thr_a']
    opt_thr_b = threshold_data['opt_thr_b']
    
    if verbose:
        print(f"Loaded optimal thresholds: DIF_a={opt_thr_a:.3f}, DIF_b={opt_thr_b:.3f}")
    
    # Setup results folder
    if results_folder is None:
        results_folder = f"{groups}_Group_Evaluation_Results"
    
    if save_results:
        # Create results folder
        os.makedirs(results_folder, exist_ok=True)
        if verbose:
            print(f"Results will be saved to: {results_folder}")
        
        # Save thresholds for reference
        threshold_df = pd.DataFrame([{
            'Threshold_a': opt_thr_a,
            'Threshold_b': opt_thr_b
        }])
        thresholds_filename = os.path.join(results_folder, f"Optimal_Thresholds_{groups}.csv")
        threshold_df.to_csv(thresholds_filename, index=False)
        
        if verbose:
            print(f"Saved thresholds to: {thresholds_filename}")
    
    # Check if data file exists
    if not os.path.exists(data_filename):
        raise FileNotFoundError(f"Data file not found: {data_filename}")
    
    # Load data
    data = pd.read_csv(data_filename)
    
    if verbose:
        print(f"Original data shape: {data.shape}")
    
    # Check feature availability and maintain order from training
    available_features = [f for f in feature_names if f in data.columns]
    missing_features = [f for f in feature_names if f not in data.columns]
    
    # Stop if any features are missing
    if missing_features:
        error_msg = f"ERROR: {len(missing_features)} required features from training are missing in data:\n"
        for feat in missing_features:
            error_msg += f"  - {feat}\n"
        error_msg += f"\nThe model requires ALL {len(feature_names)} training features to be present in the input data.\n"
        error_msg += "Please ensure your data contains all the features that were used during model training."
        raise ValueError(error_msg)
    
    if not available_features:
        raise ValueError("No matching features found between training data and input data")
    
    if verbose:
        print(f"All {len(feature_names)} required training features found in data")
        print(f"Original data shape: {data.shape}")
    
    # Check for missing data in required feature columns - STOP if any missing data found
    data_subset_features = data[feature_names]
    missing_data_info = data_subset_features.isnull()
    
    if missing_data_info.any().any():
        # Find which columns and rows have missing data
        missing_cols = missing_data_info.any(axis=0)
        missing_col_names = missing_cols[missing_cols].index.tolist()
        
        # Count missing values per column
        missing_counts = missing_data_info.sum()
        missing_counts = missing_counts[missing_counts > 0]
        
        # Find total rows with any missing data
        rows_with_missing = missing_data_info.any(axis=1).sum()
        total_missing_values = missing_data_info.sum().sum()
        
        error_msg = f"ERROR: Missing data detected in required feature columns.\n"
        error_msg += f"Total missing values: {total_missing_values}\n"
        error_msg += f"Rows affected: {rows_with_missing} out of {len(data)} total rows\n"
        error_msg += f"Columns with missing data:\n"
        
        for col in missing_col_names:
            error_msg += f"  - {col}: {missing_counts[col]} missing values\n"
        
        error_msg += f"\nThe model requires complete data for all {len(feature_names)} features.\n"
        error_msg += "Please clean your data by removing rows with missing values or imputing missing data before using this function."
        
        raise ValueError(error_msg)
    
    if verbose:
        print("✓ No missing data detected in required features")
    
    # Create DataFrame with features in the EXACT order expected by the model
    # This ensures proper feature ordering regardless of the input data column order
    # Since we verified all features are present and no missing data exists, we can safely reorder them
    data_for_scaling = data_subset_features.copy()
    
    if verbose:
        print(f"✓ Reordered features to match training model: {len(feature_names)} features")
        print(f"✓ Final data shape for model input: {data_for_scaling.shape}")
    
    # Scale the data
    X_scaled = pd.DataFrame(scaler.transform(data_for_scaling.to_numpy()))
    X_scaled.columns = feature_names
    
    # Get predictions
    if merged:
        predictions = model_dif_a.predict(X_scaled, verbose=0)
        raw_predictions_a = predictions[0]
        raw_predictions_b = predictions[1]
    else:
        raw_predictions_a = model_dif_a.predict(X_scaled, verbose=0)
        raw_predictions_b = model_dif_b.predict(X_scaled, verbose=0)
    
    # Reshape predictions
    n_samples = X_scaled.shape[0]
    n_outputs_a = len(set1_cols)
    n_outputs_b = len(set2_cols)
    
    raw_predictions_a = np.array(raw_predictions_a).reshape(n_samples, n_outputs_a)
    raw_predictions_b = np.array(raw_predictions_b).reshape(n_samples, n_outputs_b)
    
    # Create probability DataFrames
    probabilities_a = pd.DataFrame(raw_predictions_a, columns=set1_cols, index=data.index)
    probabilities_b = pd.DataFrame(raw_predictions_b, columns=set2_cols, index=data.index)
    
    if verbose:
        print(f"DIF_a probabilities shape: {probabilities_a.shape}")
        print(f"DIF_b probabilities shape: {probabilities_b.shape}")
        print(f"DIF_a probability range: [{probabilities_a.values.min():.3f}, {probabilities_a.values.max():.3f}]")
        print(f"DIF_b probability range: [{probabilities_b.values.min():.3f}, {probabilities_b.values.max():.3f}]")
    
    # Apply thresholds to get binary predictions
    binary_predictions_a = (probabilities_a > opt_thr_a).astype(int)
    binary_predictions_b = (probabilities_b > opt_thr_b).astype(int)
    
    # Combine binary predictions
    binary_predictions = pd.concat([binary_predictions_a, binary_predictions_b], axis=1)
    binary_predictions.insert(0, "Item", range(1, len(binary_predictions)+1))
    
    # Combine all probabilities
    all_probabilities = pd.concat([probabilities_a, probabilities_b], axis=1)
    all_probabilities.insert(0, "Item", range(1, len(all_probabilities)+1))
    
    # Initialize performance metrics placeholder
    performance_metrics = None
    
    # Calculate performance metrics if labels are provided
    if calculate_performance and labels_filename is not None:
        if not os.path.exists(labels_filename):
            print(f"Warning: Labels file not found: {labels_filename}. Skipping performance calculation.")
        else:
            if verbose:
                print("Calculating performance metrics...")
            
            # Load true labels
            true_labels = pd.read_csv(labels_filename)
            
            # Filter labels to match data indices (no cleaning performed, so indices should match)
            true_labels = true_labels.loc[data.index]
            
            # Get number of groups for performance calculation (similar to evaluate_models_on_test_sets)
            if len(set1_cols) > 1:
                group_ids = set()
                for label in set1_cols:
                    pair = label.split('DIF_a_')[1]
                    for g in pair.split('Group'):
                        if g:
                            group_ids.add(g)
                num_groups = len(group_ids)
            else:
                num_groups = 2
            
            if verbose:
                print(f"Calculating performance for {num_groups} groups")
            
            # Prepare binary predictions without Item column for performance calculation
            binary_preds_for_perf = pd.concat([binary_predictions_a, binary_predictions_b], axis=1)
            
            # Ensure label columns match predictions
            available_label_cols = [col for col in binary_preds_for_perf.columns if col in true_labels.columns]
            if available_label_cols:
                binary_preds_for_perf = binary_preds_for_perf[available_label_cols]
                true_labels_filtered = true_labels[available_label_cols]
                
                # Calculate performance metrics for each DIF type
                performance_results = []
                group_pairs = list(combinations(range(1, num_groups + 1), 2))
                
                for group in ['DIF_a', 'DIF_b']:
                    if num_groups > 2:
                        pred_cols = [col for col in available_label_cols if col.startswith(f'{group}_Group')]
                        true_cols = [col for col in available_label_cols if col.startswith(f'{group}_Group')]
                    else:
                        pred_cols = [col for col in available_label_cols if col.startswith(group)]
                        true_cols = [col for col in available_label_cols if col.startswith(group)]
                    
                    if pred_cols and true_cols:
                        pred_flattened = binary_preds_for_perf[pred_cols].values.flatten()
                        true_flattened = true_labels_filtered[true_cols].values.flatten()
                        
                        TP = ((pred_flattened == 1) & (true_flattened == 1)).sum()
                        FP = ((pred_flattened == 1) & (true_flattened == 0)).sum()
                        TN = ((pred_flattened == 0) & (true_flattened == 0)).sum()
                        FN = ((pred_flattened == 0) & (true_flattened == 1)).sum()
                        
                        # Calculate rates
                        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
                        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
                        
                        performance_results.append({
                            'Group': group,
                            'TP': TP,
                            'FP': FP,
                            'TN': TN,
                            'FN': FN,
                            'TPR': TPR,
                            'FPR': FPR
                        })
                
                performance_metrics = pd.DataFrame(performance_results)
                
                if verbose:
                    print("Performance Metrics:")
                    for _, row in performance_metrics.iterrows():
                        print(f"  {row['Group']}: TPR={row['TPR']:.3f}, FPR={row['FPR']:.3f}")
            else:
                print("Warning: No matching label columns found for performance calculation")
    
    # Save results if requested
    if save_results:
        # Set up results folder
        if results_folder is None:
            results_folder = f"{groups}_Group_InterDIFNet_Results"
        
        # Create results folder if it doesn't exist
        os.makedirs(results_folder, exist_ok=True)
        
        if verbose:
            print(f"Saving results to folder: {results_folder}")
        
        # Generate filenames
        if output_filename is None:
            base_name = Path(data_filename).stem
            prob_filename = f"DIF_Probabilities_{base_name}.csv"
            pred_filename = f"DIF_Predictions_{base_name}.csv"
        else:
            base_name = Path(output_filename).stem
            prob_filename = f"{base_name}_probabilities.csv"
            pred_filename = f"{base_name}_predictions.csv"
        
        # Create full paths with results folder
        prob_filepath = os.path.join(results_folder, prob_filename)
        pred_filepath = os.path.join(results_folder, pred_filename)
        
        # Save main results
        all_probabilities.to_csv(prob_filepath, index=False)
        binary_predictions.to_csv(pred_filepath, index=False)
        
        # Save performance metrics if available
        if performance_metrics is not None:
            perf_filename = f"{base_name}_performance.csv" if output_filename else f"DIF_Performance_{base_name}.csv"
            perf_filepath = os.path.join(results_folder, perf_filename)
            performance_metrics.to_csv(perf_filepath, index=False)
            print(f"Files saved: {prob_filepath}, {pred_filepath}, {perf_filepath}")
        else:
            print(f"Files saved: {prob_filepath}, {pred_filepath}")
    
    # Prepare return dictionary
    results = {
        'probabilities_a': probabilities_a,
        'probabilities_b': probabilities_b,
        'predictions': binary_predictions,
        'all_probabilities': all_probabilities,
        'model_info': {
            'groups': groups,
            'merged': merged,
            'model_type': model_metadata['model_info']['model_type'],
            'thresholds': {'opt_thr_a': opt_thr_a, 'opt_thr_b': opt_thr_b},
            'features_used': len(available_features),
            'features_total': len(feature_names)
        }
    }
    
    # Add performance metrics if calculated
    if performance_metrics is not None:
        results['performance_metrics'] = performance_metrics
    
    # Add saved file information if results were saved
    if save_results:
        results['saved_files'] = {
            'results_folder': results_folder,
            'probabilities_file': prob_filepath,
            'predictions_file': pred_filepath
        }
        if performance_metrics is not None:
            results['saved_files']['performance_file'] = perf_filepath
    
    return results

def TDC_simulation_study(groups, 
        sizes, 
        percentages=[20, 40],  
        replications=range(1, 51), 
        dif_types=['DIF_a', 'DIF_b'],
        items_to_analyze=None,
        test_thresholds=None,
        show_matrices=False,
        verbose_closure=False,
        generate_tdc_plots=False,
        generate_kmeans_plots=False,
        generate_hierarchical_plots=False,
        show_plots=False,
        print_constraints=False,
        verbose=False,
        save_mirt_constraints = False,
        save_stan_constraints=False,
        thresholds_file = None,
        show_title=True,
        color_transitive=False,
        data_folder=None,
        results_folder=None,
        random_state=12345,
        legend_fontsize=12):
    """
    Complete Transitive DIF Clustering (TDC) analysis pipeline for simulation studies.
    
    This function performs comprehensive DIF clustering analysis by loading classification results,
    applying transitive closure algorithms to identify connected group clusters, generating
    visualizations, and producing MIRT and Stan constraint syntax for psychometric modeling.
    The function processes multiple sample sizes, DIF percentages, and replications with
    organized input/output folder management.

    Parameters:
    -----------
    groups : str
        - String indicating the number of groups for analysis (e.g., "Ten", "Three", "Two").
        - Must match the naming convention used in classification result files.
    sizes : list of int
        - Sample sizes (N) to process. Each size should correspond to available classification result files with the naming pattern: 'Classification_Results_{groups}_{n}_{p}_Replication{r}.csv'
    percentages : list of int, optional
        - DIF percentages (P) representing the proportion of items with DIF to analyze. 
        - Default: [20, 40]
    replications : range or list of int, optional
        - Replication numbers (R) to process in the simulation study.
        - Default: range(1, 51)
    dif_types : list of str, optional
        - Types of Differential Item Functioning to analyze.
        - Options: ['DIF_a'] (non-uniform DIF), ['DIF_b'] (uniform DIF), or both.
        - Default: ['DIF_a', 'DIF_b']
    items_to_analyze : list of int, optional
        - Specific item indices to include in analysis. If None, analyzes all items.
        - Default: None (analyzes all items)
    test_thresholds : list of float, optional
        - DIF probability thresholds to test for clustering decisions. 
        - If None, uses default threshold range. When optimal thresholds are loaded from file, this parameter is ignored.
        - Default: None
    show_matrices : bool, optional
        - Whether to display DIF matrix heatmaps during clustering process.
        - Default: False
    verbose_closure : bool, optional
        - Whether to display detailed information about transitive closure operations.
        - Default: False
    generate_tdc_plots : bool, optional
        - Whether to generate and save TDC clustering visualization plots.
        - TDC plots are saved as: 'DIF_Clustering_Components_{groups}Groups_{dif_type}_N{n}_P{p}_R{r}_Item{item}.png'
        - Default: False
    generate_kmeans_plots : bool, optional
        - Whether to generate and save K-means clustering visualization plots.
        - K-means plots are saved as: 'K_Means_Clustering_{groups}Groups_{dif_type}_N{n}_P{p}_R{r}_Item{item}.png'
        - Uses silhouette score to determine optimal number of clusters automatically.
        - Default: False
    generate_hierarchical_plots : bool, optional
        - Whether to generate and save hierarchical clustering visualization plots.
        - Hierarchical plots are saved as: 'Hierarchical_Network_{groups}Groups_{dif_type}_N{n}_P{p}_R{r}_Item{item}.png'
        - Uses ward linkage with euclidean distance and silhouette score for optimal cluster selection.
        - Default: False
    show_plots : bool, optional
        - Whether to display plots interactively for TDC, K-means, and hierarchical clustering.
        - If False, plots are only saved but not displayed.
        - Default: False
    print_constraints : bool, optional
        - Whether to print generated MIRT constraint syntax to console.
        - Default: True
    verbose : bool, optional
        - Whether to enable detailed progress output and summaries.
        - Default: False
    save_mirt_constraints : bool, optional
        - Whether to save MIRT constraint syntax to text files. 
        - Files saved as: 'Estimated_MIRT_constraints_TDC_{groups}_{n}_{p}_Replication{r}.txt'
        - Default: False
    save_stan_constraints : bool, optional
        - Whether to save Stan constraint matrices and metadata as CSV/JSON files. 
        - Files saved as: 'Estimated_Stan_a_constraints_TDC_{groups}_{n}_{p}_Replication{r}.csv'
        - 'Estimated_Stan_b_constraints_TDC_{groups}_{n}_{p}_Replication{r}.csv'
        - 'Estimated_Stan_metadata_TDC_{groups}_{n}_{p}_Replication{r}.json'
        - Default: False
    thresholds_file : str, optional
        - Path to CSV file containing pre-computed optimal thresholds with columns 'Threshold_a' and 'Threshold_b'. 
        - If provided and file exists, uses these thresholds instead of test_thresholds.
        - Default: None
    show_title : bool, optional
        - Whether to display titles on TDC visualization plots.
        - Default: True
    color_transitive : bool, optional
        - Whether to color-code transitive vs non-transitive edges and show legend.
        - If True: original edges (black solid) and transitive edges (red dashed) with legend.
        - If False: all edges same color (black solid) with no legend.
        - Default: False
    data_folder : str, optional
        - Path to folder containing DIF classification result files. 
        - If None, loads from current working directory. 
        - Expected file naming pattern: 'Classification_Results_{groups}_{n}_{p}_Replication{r}.csv'
        - Default: None (current directory)
    results_folder : str, optional
        - Path to folder where all results will be saved. 
        - If None, creates folder named '{groups}_Group_TDC_Simulation_Study_Results'. Folder created automatically if needed.
        - Default: None
    random_state : int, optional
        - Base random seed for K-means clustering reproducibility.
        - Each replication uses random_state + replication_number to ensure proper variance across replications.
        - Only used when generate_kmeans_plots=True.
        - Default: 12345
    legend_fontsize : int, optional
        - Font size for legends in K-means and hierarchical clustering plots.
        - Default: 12

    Returns:
    --------
    None
        - Function saves results to files but returns None. Generated files include:
        - MIRT constraint syntax files (.txt) if save_mirt_constraints=True
        - Stan constraint matrices (.csv) and metadata (.json) if save_stan_constraints=True  
        - TDC clustering visualization plots (.png) if generate_tdc_plots=True
        - K-means clustering plots (.png) if generate_kmeans_plots=True
        - Hierarchical clustering plots (.png) if generate_hierarchical_plots=True

    Examples:
    --------
    # Basic TDC analysis with default settings
    >>> TDC_simulation_study(
    ...     groups="Three",
    ...     sizes=[250, 500],
    ...     percentages=[20, 40],
    ...     replications=range(1, 11),
    ...     save_mirt_constraints=True,
    ...     verbose=True
    ... )
    # Results saved to: Three_Group_TDC_Simulation_Study_Results/
    
    # Custom analysis with specific folders and options
    >>> TDC_simulation_study(
    ...     groups="Ten",
    ...     sizes=[1000, 2000],
    ...     percentages=[20],
    ...     replications=range(1, 21),
    ...     data_folder="My_Classification_Results",
    ...     results_folder="My_TDC_Analysis",
    ...     thresholds_file="optimal_thresholds.csv",
    ...     save_mirt_constraints=True,
    ...     save_stan_constraints=True,
    ...     generate_tdc_plots=True,
    ...     verbose=True
    ... )
    # Loads from: My_Classification_Results/
    # Results saved to: My_TDC_Analysis/
    
    # Analysis with both TDC and K-means clustering plots
    >>> TDC_simulation_study(
    ...     groups="Ten",
    ...     sizes=[1000],
    ...     percentages=[20],
    ...     replications=range(1, 11),
    ...     generate_tdc_plots=True,
    ...     generate_kmeans_plots=True,
    ...     results_folder="TDC_KMeans_Comparison",
    ...     verbose=True
    ... )
    # Generates and saves both TDC and K-means plots for comparison
    # Results saved to: TDC_KMeans_Comparison/
    
    # Analysis with TDC, K-means, and hierarchical clustering plots
    >>> TDC_simulation_study(
    ...     groups="Ten",
    ...     sizes=[1000],
    ...     percentages=[20],
    ...     replications=range(1, 11),
    ...     generate_tdc_plots=True,
    ...     generate_kmeans_plots=True,
    ...     generate_hierarchical_plots=True,
    ...     results_folder="All_Methods_Comparison",
    ...     verbose=True
    ... )
    # Generates and saves TDC, K-means, and hierarchical plots for comprehensive comparison
    # Results saved to: All_Methods_Comparison/

    Notes:
    ------
    - Classification result files must exist in the specified data_folder with exact naming convention
    - The function applies transitive closure to ensure clustering consistency across group comparisons
    - MIRT constraints use inclusive syntax where connected groups share parameters (no DIF)
    - Stan constraints provide constraint matrix format for Bayesian IRT modeling
    - K-means clustering uses silhouette score to automatically determine optimal number of clusters
    - Hierarchical clustering uses ward linkage with euclidean distance and silhouette score for cluster selection
    - All file operations use the results_folder for organized output management
    - Function continues processing even if individual replications fail (with warnings)
    """

    # Setup folders
    if results_folder is None:
        results_folder = f"{groups}_Group_TDC_Simulation_Study_Results"
    
    # Create results folder if it doesn't exist
    os.makedirs(results_folder, exist_ok=True)
    
    if verbose:
        if data_folder:
            print(f"Loading data from folder: {data_folder}")
        else:
            print("Loading data from current working directory")
        print(f"Results will be saved to: {results_folder}")

    # Check for and load a pre-existing threshold file
    opt_thr_a, opt_thr_b = None, None
    if thresholds_file is not None and os.path.exists(thresholds_file):
        try:
            thresholds_df = pd.read_csv(thresholds_file)
            opt_thr_a = thresholds_df.loc[0, 'Threshold_a']
            opt_thr_b = thresholds_df.loc[0, 'Threshold_b']
            print(f"Loading Constraint File {groups} Group Data")
            print(f"  DIF_a Threshold: {opt_thr_a:.4f}")
            print(f"  DIF_b Threshold: {opt_thr_b:.4f}")
        except KeyError as e:
            print(f"Error: Missing column in {thresholds_file}. Please ensure it contains 'Threshold_a' and 'Threshold_b'.")
            return
    elif thresholds_file:
        print("Warning: Threshold file not found. Defaulting to standard threshold search.")
                
    for n in sizes: 
        for p in percentages:            
            print(f"Processing - Groups: {groups}, N: {n}, DIF Percent: {p}%, Method: TDC")
            for r in replications:
        
                try:
                    dif_data = load_dif_data(groups, n, p, r, data_folder)
                    
                except Exception:
                    continue  # Skip to next replication
                
                # Perform clustering analysis
                detailed_results = {}
                
                # Perform clustering analysis for each DIF type
                for dif_type in dif_types:
                   if not dif_data[dif_type].empty:
                       if dif_type == 'DIF_a' and opt_thr_a is not None:
                           thresholds_to_use = [opt_thr_a]
                       elif dif_type == 'DIF_b' and opt_thr_b is not None:
                           thresholds_to_use = [opt_thr_b]
                       else:
                           thresholds_to_use = test_thresholds
                               
                       result = DIF_Cluster_Components_Per_Item(
                           dif_data, dif_type,
                           test_thresholds=thresholds_to_use,
                           show_matrices=show_matrices,
                           items_to_analyze=items_to_analyze,
                           verbose_closure=verbose_closure
                       )
                       detailed_results[dif_type] = result
                       # if verbose:
                       # # Create summary across items
                       #     create_summary_across_items(result, dif_type,
                       #                                 show_low_dif=show_low_dif,
                       #                                 show_high_dif=show_high_dif)

                       if generate_tdc_plots and result is not None:
                           print(f"\nGenerating TDC plots for {dif_type}...")
                           for item_idx, item_res in result['item_results'].items():
                               print(f"  Plotting Item {item_idx + 1}...")
                               # Re-extract necessary components for plotting
                               G = item_res['final_clustering']['graph']
                               cluster_dict = item_res['final_clustering']['cluster_dict']
                               groups_for_plot = item_res['groups']
                               dif_type_for_plot = item_res['dif_type']
                               original_edges = item_res['final_clustering']['original_edges']
                               transitive_edges = item_res['final_clustering']['transitive_edges']

                               visualize_connected_components_per_item(
                                   G, cluster_dict, groups_for_plot,
                                   dif_type=dif_type_for_plot, item_index=item_idx,
                                   save_plot=True, n=n, p=p, r=r, groups_name=groups,
                                   original_edges=original_edges, transitive_edges=transitive_edges,
                                   show_title=show_title, results_folder=results_folder,
                                   color_transitive=color_transitive, show_plots=show_plots,
                                   legend_fontsize=legend_fontsize
                               )
                   else:
                       print(f"Skipping analysis for {dif_type} as data is empty for N={n}, P={p}, R={r}")

                # Generate K-means clustering plots if requested
                if generate_kmeans_plots:
                    for dif_type in dif_types:
                        if not dif_data[dif_type].empty:
                            print(f"\nGenerating K-means plots for {dif_type}...")
                            df = dif_data[dif_type]
                            extracted_groups = extract_groups_from_columns(df, dif_type)
                            
                            # Determine which items to analyze
                            current_items = items_to_analyze if items_to_analyze is not None else list(range(len(df)))
                            
                            # Create results structure for K-means visualization
                            kmeans_scenario_result = {
                                'item_results': {},
                                'groups': extracted_groups,
                                'dif_type': dif_type,
                                'parameters': {'groups': groups, 'n': n, 'p': p, 'r': r}
                            }
                            
                            for item_idx in current_items:
                                print(f"  K-means plotting Item {item_idx + 1}...")
                                
                                # Create DIF matrix for this item
                                dif_matrix = create_dif_matrix_per_item(df, extracted_groups, item_idx, dif_type)
                                
                                # Apply K-means clustering with replication-specific random state
                                replication_seed = random_state + r
                                kmeans_result = apply_kmeans_to_dif_matrix(
                                    dif_matrix, extracted_groups, dif_type, item_idx,
                                    random_state=replication_seed, verbose=False
                                )
                                
                                # Store results in format expected by visualize_kmeans_network_plot
                                kmeans_scenario_result['item_results'][item_idx] = {
                                    'dif_matrix': dif_matrix,
                                    'final_clustering': {
                                        'cluster_dict': kmeans_result['cluster_dict'],
                                        'components': kmeans_result['connected_components'],
                                        'edges': kmeans_result['edges'],
                                    },
                                    'groups': extracted_groups,
                                    'dif_type': dif_type,
                                    'item_index': item_idx,
                                    'kmeans_specific': {
                                        'n_clusters': kmeans_result['n_clusters'],
                                        'silhouette_score': kmeans_result['silhouette_score'],
                                        'inertia': kmeans_result['inertia'],
                                        'features': kmeans_result['features'],
                                        'features_scaled': kmeans_result['features_scaled'],
                                        'kmeans_model': kmeans_result['kmeans_model']
                                    }
                                }
                                
                                # Generate the K-means network plot
                                visualize_kmeans_network_plot(
                                    kmeans_scenario_result, item_idx,
                                    save_plots=True, results_folder=results_folder,
                                    r=r, n=n, p=p, groups_name=groups, show_plots=show_plots,
                                    legend_fontsize=legend_fontsize
                                )

                # Generate hierarchical clustering plots if requested
                if generate_hierarchical_plots:
                    for dif_type in dif_types:
                        if not dif_data[dif_type].empty:
                            print(f"\nGenerating hierarchical plots for {dif_type}...")
                            df = dif_data[dif_type]
                            extracted_groups = extract_groups_from_columns(df, dif_type)
                            
                            # Determine which items to analyze
                            current_items = items_to_analyze if items_to_analyze is not None else list(range(len(df)))
                            
                            # Create results structure for hierarchical visualization
                            hierarchical_scenario_result = {
                                'item_results': {},
                                'groups': extracted_groups,
                                'dif_type': dif_type,
                                'parameters': {'groups': groups, 'n': n, 'p': p, 'r': r}
                            }
                            
                            for item_idx in current_items:
                                print(f"  Hierarchical plotting Item {item_idx + 1}...")
                                
                                # Create DIF matrix for this item
                                dif_matrix = create_dif_matrix_per_item(df, extracted_groups, item_idx, dif_type)
                                
                                # Apply hierarchical clustering
                                hierarchical_result = apply_hierarchical_to_dif_matrix(
                                    dif_matrix, extracted_groups, dif_type, item_idx,
                                    n_clusters=None, verbose=False
                                )
                                
                                # Store results in format expected by visualize_hierarchical_network_plot
                                hierarchical_scenario_result['item_results'][item_idx] = {
                                    'dif_matrix': dif_matrix,
                                    'final_clustering': {
                                        'cluster_dict': hierarchical_result['cluster_dict'],
                                        'components': hierarchical_result['connected_components'],
                                        'edges': hierarchical_result['edges'],
                                    },
                                    'groups': extracted_groups,
                                    'dif_type': dif_type,
                                    'item_index': item_idx,
                                    'hierarchical_specific': {
                                        'n_clusters': hierarchical_result['n_clusters'],
                                        'silhouette_score': hierarchical_result['silhouette_score'],
                                        'inertia': hierarchical_result['inertia'],
                                        'features': hierarchical_result['features'],
                                        'features_scaled': hierarchical_result['features_scaled'],
                                        'linkage_matrix': hierarchical_result['linkage_matrix'],
                                        'linkage_method': hierarchical_result['linkage_method'],
                                        'distance_metric': hierarchical_result['distance_metric']
                                    }
                                }
                                
                                # Generate the hierarchical network plot
                                visualize_hierarchical_network_plot(
                                    hierarchical_scenario_result, item_idx,
                                    save_plots=True, results_folder=results_folder,
                                    r=r, n=n, p=p, groups_name=groups, show_plots=show_plots,
                                    legend_fontsize=legend_fontsize
                                )

                # Extract connected groups (simplified)
              
                connected_results = extract_connected_groups_simple(
                    dif_data, 
                    items_to_analyze=items_to_analyze,
                    test_thresholds=[opt_thr_a, opt_thr_b] if opt_thr_a and opt_thr_b else test_thresholds
                )
                
                # Print connected groups summary
                if verbose:
                    print_connected_groups_simple(connected_results)
                
                # Generate MIRT constraints
                mirt_constraints_dict = generate_mirt_constraints(connected_results) # Use a new variable name
                
                if print_constraints:
                    mirt_constraints_string_to_print = print_mirt_constraints(mirt_constraints_dict) 
                    print(mirt_constraints_string_to_print)
                
                if save_mirt_constraints:
                    if not print_constraints:
                         mirt_constraints_string_to_print = print_mirt_constraints(mirt_constraints_dict)
                    
                    filename = f"Estimated_MIRT_constraints_TDC_{groups}_{n}_{p}_Replication{r}.txt"
                    filepath = os.path.join(results_folder, filename)
                    save_mirt_constraints_to_file(mirt_constraints_string_to_print, filepath)
                
                # Generate and save Stan constraints if requested
                if save_stan_constraints:
                    # Convert clustering results to Stan constraint format
                    group_list = extract_groups_from_columns(dif_data['DIF_a'], 'DIF_a') if not dif_data['DIF_a'].empty else extract_groups_from_columns(dif_data['DIF_b'], 'DIF_b')
                    n_items = len(dif_data['DIF_a']) if not dif_data['DIF_a'].empty else len(dif_data['DIF_b'])
                    
                    stan_constraints = create_stan_constraints_from_clustering(
                        connected_results, 
                        groups=group_list,
                        n_items=n_items
                    )
                    
                    # Export to CSV files
                    export_stan_constraints_csv(stan_constraints, groups, n, p, r, verbose=True, results_folder=results_folder)
                
                # Count items with connections
                connection_summary = {}
                
                for dif_type in dif_types:
                    if dif_type in connected_results:
                        items_data = connected_results[dif_type]['items']
                        items_with_connections = 0
                        total_connections = 0
                        
                        for item_idx, item_data in items_data.items():
                            if item_data['edges']:  # Has connections
                                items_with_connections += 1
                                total_connections += len(item_data['edges'])
                        
                        connection_summary[dif_type] = {
                            'total_items': len(items_data),
                            'items_with_connections': items_with_connections,
                            'total_connections': total_connections
                        }
                
                if verbose:
                    print("Connection Summary:")
                    for dif_type, summary in connection_summary.items():
                        print(f"  {dif_type}:")
                        print(f"    Total items analyzed: {summary['total_items']}")
                        print(f"    Items with connections: {summary['items_with_connections']}")
                        print(f"    Total group connections: {summary['total_connections']}")
                
                # Count MIRT constraints
                total_constraints = 0
                items_with_constraints = 0
                
                for item_idx, constraints in mirt_constraints_dict.items():
                    item_constraint_count = len(constraints['a1_constraints']) + len(constraints['d_constraints'])
                    if item_constraint_count > 0:
                        items_with_constraints += 1
                        total_constraints += item_constraint_count
                
                if verbose:
                    print("\nMIRT Constraint Summary:")
                    print(f"  Items requiring constraints: {items_with_constraints}")
                    print(f"  Total constraints generated: {total_constraints}")
                    
                    print("\n" + "="*80)
                    print("ANALYSIS COMPLETE")
                    print("="*80)


def TDC(classification_file, base_filename, groups,
        dif_types=['DIF_a', 'DIF_b'],
        items_to_analyze=None,
        test_thresholds=None,
        show_matrices=False,
        verbose_closure=False,
        generate_plots=True,
        print_constraints=True,
        verbose=False,
        save_mirt_constraints=False,
        save_plots=False,
        save_stan_constraints=False,
        thresholds_file=None,
        show_title=True,
        data_folder=None,
        results_folder=None,
        side_by_side_plots=False):
    """
    Complete Transitive DIF Clustering (TDC) analysis for a single classification result file.
    
    This function performs comprehensive DIF clustering analysis on a single classification
    result file, applying transitive closure algorithms to identify connected group clusters,
    generating visualizations, and producing MIRT and Stan constraint syntax for psychometric
    modeling. Results are saved to a custom folder based on the user-specified base filename.

    Parameters:
    -----------
    classification_file : str
        Name of the classification result CSV file containing DIF probabilities.
        File should contain columns following the pattern: '{DIF_type}_Item{N}_Group{M}'
        File will be loaded from data_folder if specified, otherwise from current directory.
    base_filename : str
        Base name for organizing results. Used as prefix for all output files 
        (e.g., 'Estimated_MIRT_constraints_{base_filename}.txt')
    groups : str
        String indicating the number of groups for analysis (e.g., "Ten", "Three", "Two").
        Must match the group structure in the classification file.
    dif_types : list of str, optional
        Types of Differential Item Functioning to analyze.
        Options: ['DIF_a'] (discrimination DIF), ['DIF_b'] (difficulty DIF), or both.
        Default: ['DIF_a', 'DIF_b']
    items_to_analyze : list of int, optional
        Specific item indices to include in analysis. If None, analyzes all items.
        Default: None (analyzes all items)
    test_thresholds : list of float, optional
        DIF probability thresholds to test for clustering decisions. 
        If None, uses default threshold range. Ignored when optimal thresholds loaded from file.
        Default: None
    show_matrices : bool, optional
        Whether to display DIF matrix heatmaps during clustering process.
        Default: False
    verbose_closure : bool, optional
        Whether to display detailed information about transitive closure operations.
        Default: False
    generate_plots : bool, optional
        Whether to generate and display clustering visualization plots.
        Default: True
    print_constraints : bool, optional
        Whether to print generated MIRT constraint syntax to console.
        Default: True
    verbose : bool, optional
        Whether to enable detailed progress output and summaries.
        Default: False
    save_mirt_constraints : bool, optional
        Whether to save MIRT constraint syntax to text file. 
        File saved as: 'Estimated_MIRT_constraints_{base_filename}.txt'
        Default: False
    save_plots : bool, optional
        Whether to save clustering visualization plots as PNG files. 
        Files saved as: 'DIF_Clustering_Components_{base_filename}_{dif_type}_Item{item}.png'
        When side_by_side_plots=True, files saved as: 'DIF_Clustering_Components_{base_filename}_Combined_Item{item}.png'
        Default: False
    save_stan_constraints : bool, optional
        Whether to save Stan constraint matrices and metadata as CSV/JSON files. 
        Files saved as: 'Estimated_Stan_a_constraints_{base_filename}.csv'
        'Estimated_Stan_b_constraints_{base_filename}.csv'
        'Estimated_Stan_metadata_{base_filename}.json'
        Default: False
    thresholds_file : str, optional
        Path to CSV file containing pre-computed optimal thresholds with columns 'Threshold_a' and 'Threshold_b'. 
        If provided and file exists, uses these thresholds instead of test_thresholds.
        Default: None
    show_title : bool, optional
        Whether to display titles on TDC visualization plots.
        Default: True
    data_folder : str, optional
        Path to folder containing the classification result file. 
        If None, loads from current working directory.
        Default: None (current directory)
    results_folder : str, optional
        Path to folder where all results will be saved. 
        If None, creates folder named 'TDC_Results_{base_filename}'. 
        Folder created automatically if needed.
        Default: None
    side_by_side_plots : bool, optional
        Whether to create side-by-side plots showing both DIF_a and DIF_b results for each item.
        If True, generates combined plots with DIF_a on left subplot and DIF_b on right subplot.
        If False, generates separate plots for each DIF type (default behavior).
        Default: False

    Returns:
    --------
    None
        Function saves results to files but returns None. Generated files are saved in:
        results_folder (or 'TDC_Results_{base_filename}/' if results_folder is None) and include:
        - MIRT constraint syntax file (.txt) if save_mirt_constraints=True
        - Stan constraint matrices (.csv) and metadata (.json) if save_stan_constraints=True  
        - Clustering visualization plots (.png) if save_plots=True

    Examples:
    --------
    # Basic TDC analysis with data folder
    >>> TDC(
    ...     classification_file="My_DIF_Results.csv",
    ...     base_filename="Study1_Analysis",
    ...     groups="Three",
    ...     data_folder="My_Data_Folder",
    ...     save_mirt_constraints=True,
    ...     verbose=True
    ... )
    # Loads from: My_Data_Folder/My_DIF_Results.csv
    # Results saved to: TDC_Results_Study1_Analysis/
    
    # Comprehensive analysis with custom folders
    >>> TDC(
    ...     classification_file="Classification_Results_Ten_1000_20_Replication1.csv",
    ...     base_filename="Ten_Groups_Analysis",
    ...     groups="Ten",
    ...     data_folder="Classification_Results",
    ...     results_folder="My_TDC_Results",
    ...     thresholds_file="optimal_thresholds.csv",
    ...     save_mirt_constraints=True,
    ...     save_stan_constraints=True,
    ...     save_plots=True,
    ...     verbose=True
    ... )
    # Loads from: Classification_Results/Classification_Results_Ten_1000_20_Replication1.csv
    # Results saved to: My_TDC_Results/
    
    # Analysis with side-by-side plots showing DIF_a and DIF_b together
    >>> TDC(
    ...     classification_file="My_Results.csv",
    ...     base_filename="Combined_Analysis",
    ...     groups="Three",
    ...     data_folder="My_Data",
    ...     side_by_side_plots=True,
    ...     save_plots=True,
    ...     verbose=True
    ... )
    # Creates combined plots: DIF_Clustering_Components_Combined_Analysis_Combined_ItemN.png

    Notes:
    ------
    - Classification file must exist in data_folder (or current directory) with proper naming convention
    - The function applies transitive closure to ensure clustering consistency across group comparisons
    - MIRT constraints use inclusive syntax where connected groups share parameters (no DIF)
    - Stan constraints provide constraint matrix format for Bayesian IRT modeling
    - All output files use base_filename for consistent organization
    - Function will create results folder automatically if it doesn't exist
    """

    # Setup folders
    if results_folder is None:
        results_folder = f"TDC_Results_{base_filename}"
    
    # Create results folder if it doesn't exist
    os.makedirs(results_folder, exist_ok=True)
    
    if verbose:
        if data_folder:
            print(f"Loading data from folder: {data_folder}")
        else:
            print("Loading data from current working directory")
        print(f"Classification file: {classification_file}")
        print(f"Results will be saved to: {results_folder}")

    # Check for and load a pre-existing threshold file
    opt_thr_a, opt_thr_b = None, None
    if thresholds_file is not None and os.path.exists(thresholds_file):
        try:
            thresholds_df = pd.read_csv(thresholds_file)
            opt_thr_a = thresholds_df.loc[0, 'Threshold_a']
            opt_thr_b = thresholds_df.loc[0, 'Threshold_b']
            print(f"Loading Thresholds from {thresholds_file}")
            print(f"  DIF_a Threshold: {opt_thr_a:.4f}")
            print(f"  DIF_b Threshold: {opt_thr_b:.4f}")
        except KeyError as e:
            print(f"Error: Missing column in {thresholds_file}. Please ensure it contains 'Threshold_a' and 'Threshold_b'.")
            return
    elif thresholds_file:
        print("Warning: Threshold file not found. Defaulting to standard threshold search.")
                
    try:
        # Construct full file path
        if data_folder:
            file_path = os.path.join(data_folder, classification_file)
        else:
            file_path = classification_file
            
        # Load single classification file
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Classification file not found: {file_path}")
        
        # Load the classification data using the same parsing logic as load_dif_data
        data = pd.read_csv(file_path, index_col=0)
        
        # Parse data into DIF_a and DIF_b DataFrames (same logic as load_dif_data)
        dif_a_cols = [col for col in data.columns if col.startswith('DIF_a')]
        dif_b_cols = [col for col in data.columns if col.startswith('DIF_b')]
        
        dif_data = {
            'DIF_a': data[dif_a_cols] if dif_a_cols else pd.DataFrame(),
            'DIF_b': data[dif_b_cols] if dif_b_cols else pd.DataFrame()
        }
        
        if verbose:
            print(f"Loaded classification data:")
            print(f"  DIF_a columns: {len(dif_a_cols)}")
            print(f"  DIF_b columns: {len(dif_b_cols)}")
            print(f"  Total rows: {len(data)}")
        
    except Exception as e:
        print(f"Error loading classification file: {e}")
        return
    
    print(f"Processing - Groups: {groups}, Method: TDC, File: {base_filename}")
    
    # Perform clustering analysis
    detailed_results = {}
    
    # Perform clustering analysis for each DIF type
    for dif_type in dif_types:
       if not dif_data[dif_type].empty:
           if dif_type == 'DIF_a' and opt_thr_a is not None:
               thresholds_to_use = [opt_thr_a]
           elif dif_type == 'DIF_b' and opt_thr_b is not None:
               thresholds_to_use = [opt_thr_b]
           else:
               thresholds_to_use = test_thresholds
                   
           result = DIF_Cluster_Components_Per_Item(
               dif_data, dif_type,
               test_thresholds=thresholds_to_use,
               show_matrices=show_matrices,
               items_to_analyze=items_to_analyze,
               verbose_closure=verbose_closure
           )
           detailed_results[dif_type] = result

           if generate_plots and result is not None and not side_by_side_plots:
               print(f"\nGenerating plots for {dif_type}...")
               for item_idx, item_res in result['item_results'].items():
                   print(f"  Plotting Item {item_idx + 1}...")
                   # Re-extract necessary components for plotting
                   G = item_res['final_clustering']['graph']
                   cluster_dict = item_res['final_clustering']['cluster_dict']
                   dif_type_for_plot = item_res['dif_type']
                   original_edges = item_res['final_clustering']['original_edges']
                   transitive_edges = item_res['final_clustering']['transitive_edges']

                   # Custom visualization function for single-file TDC
                   visualize_connected_components_per_item_tdc(
                       G, cluster_dict,
                       dif_type=dif_type_for_plot, item_index=item_idx,
                       save_plot=save_plots, base_filename=base_filename,
                       original_edges=original_edges, transitive_edges=transitive_edges,
                       show_title=show_title, results_folder=results_folder
                   )
       else:
           if not side_by_side_plots:
               print(f"Skipping analysis for {dif_type} as data is empty")

    # Generate side-by-side plots if requested
    if generate_plots and side_by_side_plots:
        print("\nGenerating side-by-side plots combining DIF_a and DIF_b results...")
        
        # Get all items that were analyzed in either DIF type
        all_items = set()
        for dif_type in dif_types:
            if dif_type in detailed_results and detailed_results[dif_type] is not None:
                all_items.update(detailed_results[dif_type]['item_results'].keys())
        
        for item_idx in sorted(all_items):
            print(f"  Plotting combined results for Item {item_idx + 1}...")
            visualize_connected_components_side_by_side_tdc(
                detailed_results, item_idx, base_filename,
                save_plot=save_plots, show_title=show_title, results_folder=results_folder
            )

    # Extract connected groups (simplified)
    connected_results = extract_connected_groups_simple(
        dif_data, 
        items_to_analyze=items_to_analyze,
        test_thresholds=[opt_thr_a, opt_thr_b] if opt_thr_a and opt_thr_b else test_thresholds
    )
    
    # Print connected groups summary
    if verbose:
        print_connected_groups_simple(connected_results)
    
    # Generate MIRT constraints
    mirt_constraints_dict = generate_mirt_constraints(connected_results)
    
    if print_constraints:
        mirt_constraints_string_to_print = print_mirt_constraints(mirt_constraints_dict) 
        print(mirt_constraints_string_to_print)
    
    if save_mirt_constraints:
        if not print_constraints:
             mirt_constraints_string_to_print = print_mirt_constraints(mirt_constraints_dict)
        
        filename = f"Estimated_MIRT_constraints_{base_filename}.txt"
        filepath = os.path.join(results_folder, filename)
        save_mirt_constraints_to_file(mirt_constraints_string_to_print, filepath)
    
    # Generate and save Stan constraints if requested
    if save_stan_constraints:
        # Convert clustering results to Stan constraint format
        group_list = extract_groups_from_columns(dif_data['DIF_a'], 'DIF_a') if not dif_data['DIF_a'].empty else extract_groups_from_columns(dif_data['DIF_b'], 'DIF_b')
        n_items = len(dif_data['DIF_a']) if not dif_data['DIF_a'].empty else len(dif_data['DIF_b'])
        
        stan_constraints = create_stan_constraints_from_clustering(
            connected_results, 
            groups=group_list,
            n_items=n_items
        )
        
        # Export to CSV files with base_filename
        export_stan_constraints_csv_tdc(stan_constraints, base_filename, verbose=True, results_folder=results_folder)
    
    # Count items with connections
    connection_summary = {}
    
    for dif_type in dif_types:
        if dif_type in connected_results:
            items_data = connected_results[dif_type]['items']
            items_with_connections = 0
            total_connections = 0
            
            for item_idx, item_data in items_data.items():
                if item_data['edges']:  # Has connections
                    items_with_connections += 1
                    total_connections += len(item_data['edges'])
            
            connection_summary[dif_type] = {
                'total_items': len(items_data),
                'items_with_connections': items_with_connections,
                'total_connections': total_connections
            }
    
    if verbose:
        print("Connection Summary:")
        for dif_type, summary in connection_summary.items():
            print(f"  {dif_type}:")
            print(f"    Total items analyzed: {summary['total_items']}")
            print(f"    Items with connections: {summary['items_with_connections']}")
            print(f"    Total group connections: {summary['total_connections']}")
    
    # Count MIRT constraints
    total_constraints = 0
    items_with_constraints = 0
    
    for item_idx, constraints in mirt_constraints_dict.items():
        item_constraint_count = len(constraints['a1_constraints']) + len(constraints['d_constraints'])
        if item_constraint_count > 0:
            items_with_constraints += 1
            total_constraints += item_constraint_count
    
    if verbose:
        print("\nMIRT Constraint Summary:")
        print(f"  Items requiring constraints: {items_with_constraints}")
        print(f"  Total constraints generated: {total_constraints}")
        
        print("\n" + "="*80)
        print("TDC ANALYSIS COMPLETE")
        print("="*80)
            