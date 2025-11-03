import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import umap
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from config.binn_config import BinnConfig
from config.gnn_binn_config import GNNBinnConfig
from binn.src.data_loader import load_initial_data


def print_top_k_nodes_per_cluster(cluster_dfs, dfs, gnn_binn_config):
    """
    Analyzes node scores within each cluster, formats the node names,
    and prints the top k nodes for each layer with directionality.

    Args:
        cluster_dfs (dict): A dictionary where keys are cluster IDs and values are
                            DataFrames containing the samples for that cluster.
        dfs (dict): A dictionary where keys are layer names (e.g., 'hidden_0') and values
                    are the original, unprocessed score DataFrames.
        gnn_binn_config: A configuration object containing the 'top_k_nodes' attribute.
    """
    print("\n--- Top K Nodes Analysis per Cluster ---")
    top_k = gnn_binn_config.top_k_nodes

    # Iterate through each cluster's DataFrame
    for cluster_id, cluster_df in cluster_dfs.items():
        print(f"\n=========================================")
        print(f"         Analyzing Cluster {cluster_id}            ")
        print(f"=========================================")

        # Get the original sample indices for the current cluster
        sample_indices = cluster_df.index

        # Iterate through each layer we have original scores for
        for layer_name in sorted(dfs.keys()):
            # Get the original scores for the samples in the current cluster
            layer_df_for_cluster = dfs[layer_name].loc[sample_indices]

            # 1. Calculate the mean for each node
            mean_scores = layer_df_for_cluster.mean()

            # 2. Get absolute mean values for sorting
            abs_mean_scores = np.abs(mean_scores)

            # 3. Sort nodes by absolute mean to find the top k
            top_nodes_series = abs_mean_scores.sort_values(ascending=False).head(top_k)

            print(f"\n  Top {top_k} nodes for layer '{layer_name}':")

            if top_nodes_series.empty:
                print("    No nodes to display for this layer.")
                continue

            # Display the results
            for i, (node, _) in enumerate(top_nodes_series.items()):
                original_mean = mean_scores[node]
                
                # Determine directionality sign (pos/neg) from the mean score
                direction = "pos" if original_mean > 0 else "neg"
                
                # Format the node name using the abbreviation function
                formatted_name = format_node_label(node)
                
                # Print the result in a single, clean line
                print(f"    {i+1}. ({direction}) {formatted_name:<45} | Mean: {original_mean: 9.4f}")


def compare_clusters(cluster_dfs, dfs, gnn_binn_config):
    """
    Compares cluster 0 and cluster 1 by generating scatter plots for each layer.
    Each point is a node, x=mean importance in cluster 0, y=mean in cluster 1.
    Labels only nodes with high difference (|mean_0 - mean_1| top 10).

    Args:
        cluster_dfs (dict): Cluster-specific DataFrames (keys: 0, 1).
        dfs (dict): Original score DataFrames per layer.
        gnn_binn_config: Config for top_k_nodes (used for # of labels) and plot_save_path.
    """
    if 0 not in cluster_dfs or 1 not in cluster_dfs:
        print("Skipping comparison: Clusters 0 and/or 1 not found.")
        return

    print("\n--- Comparing Cluster 0 vs Cluster 1 ---")
    num_labels = gnn_binn_config.top_k  # Reuse top_k for # of labeled points; can change if needed

    # Get sample indices for each cluster
    sample_indices_0 = cluster_dfs[0].index
    sample_indices_1 = cluster_dfs[1].index

    # Iterate over each layer
    for layer_name in sorted(dfs.keys()):
        # Compute means for each cluster
        mean_0 = dfs[layer_name].loc[sample_indices_0].mean()
        mean_1 = dfs[layer_name].loc[sample_indices_1].mean()

        # Create figure
        plt.figure(figsize=(10, 8))
        
        plt.scatter(mean_0, mean_1, alpha=0.6, s=20, color='#457B9D')  # Points for all nodes

        # Add diagonal reference line
        min_val = min(mean_0.min(), mean_1.min())
        max_val = max(mean_0.max(), mean_1.max())
        
        plt.plot([min_val, max_val], [min_val, max_val], color='darkgray', linestyle='--')

        # Compute differences and select top for labeling
        diffs = np.abs(mean_0 - mean_1)
        top_diff_nodes = diffs.sort_values(ascending=False).head(num_labels).index

        # Label high-difference nodes
        for node in top_diff_nodes:
            x = mean_0[node]
            y = mean_1[node]
            formatted_name = format_node_label(node)
            # NEW: Split into words and insert \n every 3 words for multi-line labels
            words = formatted_name.split()
            wrapped_label = '\n'.join([' '.join(words[i:i+3]) for i in range(0, len(words), 3)])
            plt.text(x, y, wrapped_label, fontsize=9, ha='right', va='bottom')

        # Labels and title
        # plt.xlabel('Mean Importance (Cluster 0)')
        # plt.ylabel('Mean Importance (Cluster 1)')
        plt.title(f'Cluster Comparison: Layer {layer_name}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

        # Save plot
        plot_filename = f"scatter_comparison_layer_{layer_name}.png"
        save_path = os.path.join(gnn_binn_config.plot_save_path, plot_filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved scatter plot for layer '{layer_name}' to {save_path}")


def main():
    binn_config = BinnConfig()
    gnn_binn_config = GNNBinnConfig()

    # Load the original data to get the true labels
    data, _, _ = load_initial_data(binn_config)
    y = data['y']


   # Ensure the plot save directory exists
    os.makedirs(gnn_binn_config.plot_save_path, exist_ok=True)


    #  Load and Filter Score Files 
    all_scores_files = glob.glob(os.path.join(gnn_binn_config.scores_dir, "stable_sample_scores_*.csv"))
    
    # Load all relevant dataframes into a dictionary, ignoring 'end_to_end'
    dfs = {}
    for file_path in all_scores_files:
        layer_name = os.path.basename(file_path).replace("stable_sample_scores_", "").replace(".csv", "")
        if 'end_to_end' in layer_name:
            continue
        print(f"Loading scores for layer: {layer_name}")
        dfs[layer_name] = pd.read_csv(file_path, index_col="sample_id")




    # Select DataFrames Step 1 - Create a DataFrame with ALL features for later analysis 
    print("Creating a full feature matrix for post-clustering analysis...")
    all_available_layers = sorted([layer for layer in dfs if layer.startswith('hidden_')], key=lambda x: int(x.split('_')[1]))
    
    full_dfs_to_combine = []
    for layer_name in all_available_layers:
        df = dfs[layer_name].copy()
        df.columns = [f"{layer_name}__{col}" for col in df.columns]
        full_dfs_to_combine.append(df)
    
    full_feature_df = pd.concat(full_dfs_to_combine, axis=1)
    print(f"Full feature matrix shape for analysis: {full_feature_df.shape}")

    # Select a SUBSET of features specifically for clustering 
    print(f"\nSelecting features for clustering based on config: '{gnn_binn_config.layers_to_combine}'")
    layers_for_clustering = []
    if gnn_binn_config.layers_to_combine == 'gene':
        if 'hidden_0' in dfs: layers_for_clustering.append('hidden_0')
    elif gnn_binn_config.layers_to_combine == 'go':
        layers_for_clustering = sorted([layer for layer in dfs if layer.startswith('hidden_') and layer != 'hidden_0'], key=lambda x: int(x.split('_')[1]))
    elif gnn_binn_config.layers_to_combine == 'combined':
        layers_for_clustering = all_available_layers
    
    if not layers_for_clustering:
        raise ValueError(f"No score files found for clustering mode: '{gnn_binn_config.layers_to_combine}'")

    clustering_prefixes = tuple([f"{layer}__" for layer in layers_for_clustering])
    clustering_cols = [col for col in full_feature_df.columns if col.startswith(clustering_prefixes)]
    
    clustering_feature_df = full_feature_df[clustering_cols]
    print(f"Feature matrix shape for clustering: {clustering_feature_df.shape}")



    # Preprocessing and Dimensionality Reduction 
    # Preprocessing: Take absolute values and then L2 normalize per sample
    scores = clustering_feature_df.values
    abs_scores = np.abs(scores)
    normalized_scores = normalize(abs_scores, norm='l2', axis=1)

    print(f"Performing {gnn_binn_config.dim_reduction_method.upper()} on the combined feature set...")
    if gnn_binn_config.dim_reduction_method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=32, min_dist=0.0)
        embeddings = reducer.fit_transform(normalized_scores)

    elif gnn_binn_config.dim_reduction_method == 'tsne':
        # print(r"Applying PCA first to preserve 95% of the variance...")

        # 1. Initialize and fit PCA
        pca = PCA(n_components=200, random_state=42)
        # Use the PCA-transformed data as input for t-SNE
        data_for_tsne = pca.fit_transform(normalized_scores)

        # Assess how many components were selected
        num_components = pca.n_components_
        print(f"PCA selected {num_components} components.")


        reducer = TSNE(n_components=2, random_state=42, perplexity=5)
        embeddings = reducer.fit_transform(data_for_tsne)





    # Plot the Final Result 
    print("Generating plot...")
    plt.figure(figsize=(12, 10))

    # Define colors and labels
    colors = {'CRPR': '#812122', 'primary cancer': '#B5B5B5'}
    # Assuming y contains 1 for CRPR and 0 for Primary Cancer
    # Create a list of colors for each point
    point_colors = ['#812122' if label == 1 else '#B5B5B5' for label in y]

    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=point_colors, alpha=0.9)

    # Create a legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='CRPR',
                            markerfacecolor=colors['CRPR'], markersize=10),
                    Line2D([0], [0], marker='o', color='w', label='primary cancer',
                            markerfacecolor=colors['primary cancer'], markersize=10)]

    plt.legend(handles=legend_elements, fontsize=13)

    # plt.title(f"using {gnn_binn_config.layers_to_combine} layer scores")

    # Use a filename that reflects the combination mode
    plot_filename = f"umap_{gnn_binn_config.layers_to_combine}.png"
    labels_plot_path = os.path.join(gnn_binn_config.plot_save_path, plot_filename)

    plt.savefig(labels_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved combined labels plot to {labels_plot_path}")



    # Clustering 
    print(f"Performing {gnn_binn_config.clustering_method.upper()} clustering...")

    if gnn_binn_config.clustering_method == 'kmeans':
        n_clusters = gnn_binn_config.n_clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        print(f"K-Means finished with {n_clusters} clusters.")

    elif gnn_binn_config.clustering_method == 'dbscan':
        # DBSCAN requires careful tuning of eps and min_samples
        dbscan = DBSCAN(eps=0.34, min_samples=15)
        cluster_labels = dbscan.fit_predict(embeddings)

        # Calculate number of clusters and noise points for reporting
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        # number of each cluster
        cluster_sizes = [list(cluster_labels).count(i) for i in range(n_clusters)]
        print(f"Cluster sizes (excluding noise): {cluster_sizes}")
        n_noise = list(cluster_labels).count(-1)
        print(f"DBSCAN finished. Estimated clusters: {n_clusters}; Noise points: {n_noise}")



    # Create and Save Debug Plot for Clusters 
    # This unified plotter handles both standard K-Means clusters 
    # and DBSCAN's potential -1 'noise' labels.
    print("Generating debug plot for clusters...")
    plt.figure(figsize=(12, 10))

    # Get unique labels to handle colors and legend
    unique_labels = sorted(list(set(cluster_labels)))

    custom_colors = [
        '#457B9D',
        '#812122',
        '#E09F3E'
    ]


    for i, label in enumerate(unique_labels):
        # Create a mask for points belonging to the current label
        mask = (cluster_labels == label)
        
        if label == -1:
            # Special styling for DBSCAN noise
            plt.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                        c='lightgray', marker='x', s=30, label='noise', alpha=0.5)
        else:
            # Standard styling for clusters
            plt.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                        color=custom_colors[i % len(custom_colors)],
                        marker='o', s=50, label=f'cluster {label}', alpha=0.9, edgecolors='k', linewidth=0.2)

    plt.legend(loc='best')

    cluster_plot_path = os.path.join(gnn_binn_config.plot_save_path, f"{gnn_binn_config.clustering_method}_clusters_{gnn_binn_config.layers_to_combine}.png")
    plt.savefig(cluster_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved cluster debug plot to {cluster_plot_path}")




    # Separate Data into DataFrames per Cluster 
    print("\nSeparating data by cluster using the full feature set...")
    full_feature_df['cluster'] = cluster_labels # Apply labels to the complete dataframe
    
    cluster_dfs = {}
    for i in range(n_clusters):
        # Create cluster-specific dataframes from the FULL set of features
        cluster_df = full_feature_df[full_feature_df['cluster'] == i].drop(columns=['cluster'])
        cluster_dfs[i] = cluster_df
        print(f"Cluster {i} contains {len(cluster_df)} samples.")


    # Print Top K Nodes per Cluster
    print_top_k_nodes_per_cluster(cluster_dfs, dfs, gnn_binn_config)






if __name__ == "__main__":
    main()