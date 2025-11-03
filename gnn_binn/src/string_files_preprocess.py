import json
import numpy as np
import pandas as pd
import os


def map_proteins_to_genes(
    protein_nodes_path,
    ppi_edges_path,
    mapping_file_path,
    gene_nodes_output_path,
    ggi_edges_output_path,
):
    # maps protein-protein interactions to gene-gene interactions.

    # create output paths if they don't exist
    os.makedirs(os.path.dirname(gene_nodes_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(ggi_edges_output_path), exist_ok=True)

    # load protein data
    with open(protein_nodes_path, "r") as f:
        protein_nodes = json.load(f)
    ppi_edges = np.load(ppi_edges_path)
    protein_map_df = pd.read_csv(mapping_file_path, sep="\t")

    # drop rows where the UniProt ID is missing
    protein_map_df = protein_map_df.dropna(subset=["UniProtKB/Swiss-Prot ID"])

    # create a mapping from protein ID to UniProt ID
    protein_to_uniprot = pd.Series(
        protein_map_df["UniProtKB/Swiss-Prot ID"].values,
        index=protein_map_df["Protein stable ID"],
    ).to_dict()

    # create new gene nodes and edges
    gene_gene_edges = set()

    # invert the protein_nodes dictionary for easy lookup
    index_to_protein = {v: k for k, v in protein_nodes.items()}

    for edge in ppi_edges:
        protein1_id = index_to_protein.get(edge[0])
        protein2_id = index_to_protein.get(edge[1])

        if protein1_id in protein_to_uniprot and protein2_id in protein_to_uniprot:
            gene1_id = protein_to_uniprot[protein1_id]
            gene2_id = protein_to_uniprot[protein2_id]

            # ensure consistent edge direction to handle duplicates
            if gene1_id and gene2_id and gene1_id != gene2_id:
                sorted_edge = tuple(sorted((gene1_id, gene2_id)))
                gene_gene_edges.add(sorted_edge)

    # create a mapping from gene ID to a new index
    unique_genes = sorted(list(set(g for e in gene_gene_edges for g in e)))
    gene_nodes = {gene: i for i, gene in enumerate(unique_genes)}

    # convert edges to use the new indices
    indexed_ggi_edges = [
        (gene_nodes[edge[0]], gene_nodes[edge[1]]) for edge in gene_gene_edges
    ]

    # save the new files
    with open(gene_nodes_output_path, "w") as f:
        json.dump(gene_nodes, f, indent=2)
    np.save(ggi_edges_output_path, np.array(indexed_ggi_edges))

    print(f"Created {gene_nodes_output_path} with {len(gene_nodes)} unique genes.")
    print(f"Created {ggi_edges_output_path} with {len(indexed_ggi_edges)} unique gene-gene interactions.")

    return unique_genes


def filter_and_order_genes(
    gene_nodes_path,
    ggi_edges_path,
    available_and_ordered_uniprot_ids,  
    filtered_ordered_nodes_path,
    filtered_ordered_edges_path,
):
    # filters genes based on the available data and orders them according to BINN network's gene layer.

    # load gene data generated from the previous step
    with open(gene_nodes_path, "r") as f:
        gene_nodes = json.load(f)  
    ggi_edges = np.load(ggi_edges_path)

    #  create a set of genes that are in our STRING-based graph
    genes_in_graph = set(gene_nodes.keys())
    
    # create a set from the provided list for efficient filtering
    genes_in_binn_list = set(available_and_ordered_uniprot_ids)

    # find the intersection: genes that are BOTH in our graph AND in the BINN/data list
    genes_to_keep = genes_in_graph.intersection(genes_in_binn_list)

    # create the new ordered node mapping
    final_ordered_nodes = {
        uniprot_id: i
        for i, uniprot_id in enumerate(available_and_ordered_uniprot_ids)
        if uniprot_id in genes_to_keep
    }

    # invert the original gene_nodes for lookup by index
    index_to_gene = {v: k for k, v in gene_nodes.items()}

    # filter and re-index the edges
    final_edges = []
    for edge in ggi_edges:
        gene1 = index_to_gene.get(edge[0])
        gene2 = index_to_gene.get(edge[1])

        # keep the edge only if both genes are in our final node list
        if gene1 in final_ordered_nodes and gene2 in final_ordered_nodes:
            new_index1 = final_ordered_nodes[gene1]
            new_index2 = final_ordered_nodes[gene2]
            final_edges.append((new_index1, new_index2))

    # save the final files
    with open(filtered_ordered_nodes_path, "w") as f:
        json.dump(final_ordered_nodes, f, indent=2)
    np.save(filtered_ordered_edges_path, np.array(final_edges))

    print(f"Created {filtered_ordered_nodes_path} with {len(final_ordered_nodes)} filtered and ordered genes.")
    print(f"Created {filtered_ordered_edges_path} with {len(final_edges)} filtered and ordered edges.")