from numpy import unique
import obonet
import networkx as nx
from typing import Dict, List, Optional

import pandas as pd

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from binn.src.model.process_gaf_file import *


class PathwayNetwork:
    """
    Represents a pathway network, derived from an ontology like GO.

    Attributes:
        graph (nx.MultiDiGraph): The raw networkx graph loaded from the OBO file.
        id_to_name (Dict[str, str]): Mapping from node ID to human-readable name.
        name_to_id (Dict[str, str]): Mapping from human-readable name to node ID.
        node_to_index (Dict[str, int]): Mapping from node ID to sequential index.
        index_to_node (Dict[int, str]): Mapping from sequential index to node ID.
        layer_indices (List[int]): List where layer_indices[i] is the starting
                                   sequential index for hierarchy level i.
        edge_index (Optional[List[List[int]]]): Edge list in [source_idx, target_idx]
                                                format using sequential indices.
                                                None until calculated.
        max_level (Optional[int]): Maximum level the graph is pruned to.
        max_hierarchy_level (Optional[int]): Maximum hierarchy level in the graph.
        protein_input_nodes (pd.DataFrame): DataFrame containing input protein nodes.
        added_protein_node_ids (set): Set of protein node IDs added to the graph.
        excluded_protein_features (set): Set of protein features that were not added.
        gaf_is_preprocessed (bool): Indicates if the GAF file has been preprocessed.
        gaf_file_path (Optional[str]): Path to the GAF file.
        preprocessed_gaf_file_path (Optional[str]): Path to the preprocessed GAF file.
        verbose (bool): If True, print progress and statistics during processing.

    """

    def __init__(self, obo_path: str,
                 protein_input_nodes: pd.DataFrame,
                 root_nodes_to_include: Optional[List[str]] = None,
                 gaf_is_preprocessed: bool = False,
                 gaf_file_path: Optional[str] = '',
                 preprocessed_gaf_file_path: Optional[str] = '',
                 available_nodes_in_string: Optional[List[str]] = None,
                 available_features_in_data: Optional[List[str]] = None,
                 max_level: Optional[int] = 10,
                 verbose: bool = False):
        """
        Initializes the PathwayNetwork by loading the graph from an OBO file.

        Args:
            obo_path (str): Path to the OBO file (e.g., 'go-basic.obo').
            verbose (bool): If True, print progress and statistics during processing.
        """
        print(f"Loading graph from: {obo_path}")
        self.graph: nx.MultiDiGraph = obonet.read_obo(obo_path)
        print(f"Graph loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")

        self.verbose: bool = verbose

        # Initialize properties that will be calculated
        self.id_to_name: Dict[str, str] = {id_: data.get('name', id_) for id_, data in self.graph.nodes(data=True)}
        self.name_to_id: Dict[str, str] = {name: id_ for id_, name in self.id_to_name.items()}
        self.node_to_index: Dict[str, int] = {}
        self.index_to_node: Dict[int, str] = {}
        self.layer_indices: List[int] = []
        self.edge_index: Optional[List[List[int]]] = None

        self.max_level: Optional[int] = max_level
        self.max_hierarchy_level: Optional[int] = None

        self.protein_input_nodes = protein_input_nodes
        self.added_protein_node_ids = set()
        self.excluded_protein_features = set()

        self.root_nodes_to_include = root_nodes_to_include

        self.gaf_is_preprocessed = gaf_is_preprocessed
        self.gaf_file_path = gaf_file_path
        self.preprocessed_gaf_file_path = preprocessed_gaf_file_path
        self.available_nodes_in_string = available_nodes_in_string
        self.available_features_in_data = available_features_in_data

        self._build_network()

    def _prune_using_root_nodes(self, root_nodes_to_include: List[str]) -> None:
        """
        Prune the graph to include only the specified root nodes and their descendants.

        Args:
            root_nodes (List[str]): List of root node names to include in the pruned graph.
        """
        if self.verbose:
            print(f"\nPruning graph to include only specified root nodes: {len(root_nodes_to_include)}")

        
        if root_nodes_to_include is None or len(root_nodes_to_include) == 0:
            print("No root nodes specified to include, using all three.")
            root_nodes_to_include = [node for node, out_degree in self.graph.out_degree() if out_degree == 0]
        
        if len(root_nodes_to_include) == 3:
            return

        # Determine root node IDs from names
        root_nodes_to_include = [self.name_to_id.get(name, name) for name in root_nodes_to_include]

        nodes_before = self.graph.number_of_nodes()
        edges_before = self.graph.number_of_edges()

        # Find all nodes that have paths to the specified root nodes
        nodes_to_keep = set(root_nodes_to_include)

        # BFS from the root nodes
        queue = deque(root_nodes_to_include)

        while queue:
            current = queue.popleft()

            # Get all predecessors (nodes that point to current)
            for predecessor in self.graph.predecessors(current):
                if predecessor not in nodes_to_keep:
                    nodes_to_keep.add(predecessor)
                    queue.append(predecessor)
        
        nodes_to_remove = [node for node in self.graph.nodes() if node not in nodes_to_keep]

        if self.verbose:
            print(f"Keeping {len(nodes_to_keep)} nodes connected to specified root nodes.")
            print(f"Removing {len(nodes_to_remove)} nodes not connected to specified root nodes.")

        # Remove nodes not connected to the specified root nodes
        self.graph.remove_nodes_from(nodes_to_remove)

        nodes_after = self.graph.number_of_nodes()
        edges_after = self.graph.number_of_edges()

        if self.verbose:
            print(f"Pruning complete.")
            print(f"Nodes: {nodes_before} -> {nodes_after} ({nodes_before - nodes_after} removed)")
            print(f"Edges: {edges_before} -> {edges_after} ({edges_before - edges_after} removed)")

    def _calculate_hierarchy_levels(self) -> None:
        """
        Calculates and adds the 'hierarchy_level' attribute to each node.

        Uses maximum distance from root nodes (nodes with out_degree 0).
        Root nodes are level 0. A node's level is max(parent_levels) + 1.
        Updates `self.max_hierarchy_level`.
        """
        if self.verbose:
            print("\nCalculating hierarchy levels...")

        # Find root nodes (nodes with no outgoing edges)
        root_nodes = [node for node, out_degree in self.graph.out_degree() if out_degree == 0]

        if self.verbose:
            print(f"Found {len(root_nodes)} root nodes (terms with no children):")
            for node in root_nodes[:5]:  # Print only a few examples
                print(f"• {node}: {self.id_to_name.get(node, 'N/A')}")

        # Initialize all nodes with level -1 (unprocessed)
        for node in self.graph.nodes():
            self.graph.nodes[node]['hierarchy_level'] = -1

        # Set root nodes to level 0
        for root in root_nodes:
            self.graph.nodes[root]['hierarchy_level'] = 0

        # Process nodes level by level
        current_level = 0
        nodes_at_current_level = set(root_nodes)
        processed_nodes = set(root_nodes)
        max_level_found = 0

        # Continue until we've processed all nodes or no more nodes at current level
        while nodes_at_current_level:
            if self.verbose and current_level % 5 == 0:
                print(f"Processing {len(nodes_at_current_level)} nodes at level {current_level}")

            next_level_nodes = set()

            # Process all nodes at the current level
            for node in nodes_at_current_level:

                # Find all children of this node
                for child in self.graph.predecessors(node):
                    # Calculate the new level for this child
                    new_level = current_level + 1           
                    max_level_found = max(max_level_found, new_level)

                    # If the child is unprocessed or the new level is higher, update it
                    child_current_level = self.graph.nodes[child]['hierarchy_level']
                    if child_current_level == -1 or new_level > child_current_level:
                        self.graph.nodes[child]['hierarchy_level'] = new_level
                        next_level_nodes.add(child)

                    # Mark as processed either way
                    processed_nodes.add(child)

            # Move to the next level
            current_level += 1
            nodes_at_current_level = next_level_nodes

        self.max_hierarchy_level = max_level_found

        # Check for any nodes that weren't processed (this shouldn't happen)
        unprocessed = [node for node, data in self.graph.nodes(data=True) if data['hierarchy_level'] == -1]
        if unprocessed and self.verbose:
            print(f"Warning: {len(unprocessed)} nodes were not reached from any root node.")

        if self.verbose:
            print(f"Hierarchy calculation complete. Max level: {self.max_hierarchy_level}")
            level_counts = {}
            for node in self.graph.nodes():
                level = self.graph.nodes[node]['hierarchy_level']
                level_counts[level] = level_counts.get(level, 0) + 1
            print("Nodes per hierarchy level:")
            for level in sorted(level_counts.keys()):
                print(f"  Level {level}: {level_counts[level]} nodes")

    def _calculate_sequential_indices(self) -> None:
        """
        Assigns 'sequential_index' to nodes based on hierarchy level.
        """
        
        if self.verbose:
            print("\nCalculating sequential indices...")

        nodes_by_level: Dict[int, List[str]] = {level: [] for level in range(0, self.max_hierarchy_level + 1)}
        for node, data in self.graph.nodes(data=True):
            level = data.get('hierarchy_level', -1)
            nodes_by_level[level].append(node)

        current_index = 0
        self.node_to_index = {}
        self.index_to_node = {}
        # Start layer_indices with 0 for level 0
        self.layer_indices = [0] * (self.max_hierarchy_level + 1)
        processed_levels = 0

        # Assign indices level by level, starting from 0
        for level in range(self.max_hierarchy_level + 1):
            nodes = sorted(nodes_by_level[level])  # Sort for reproducibility

            self.layer_indices[level] = current_index
            processed_levels += 1

            for node in nodes:
                self.graph.nodes[node]['sequential_index'] = current_index
                self.node_to_index[node] = current_index
                self.index_to_node[current_index] = node
                current_index += 1

        # Add the total number of indexed nodes as the final boundary
        # Adjust layer_indices length if needed
        self.layer_indices = self.layer_indices[:processed_levels]
        self.layer_indices.append(current_index)

        if self.verbose:
            print(f"Sequential indices assigned. Total indexed nodes: {current_index}")
            print(f"Layer start indices: {self.layer_indices}")
            print("\nExample indices:")
            for level in range(min(3, self.max_hierarchy_level + 1)):
                nodes = nodes_by_level.get(level, [])[:3]
                if nodes:
                    print(f" Level {level}:")
                    for node in nodes:
                        idx = self.node_to_index[node]
                        print(f"  • {node} (Index {idx}): {self.id_to_name.get(node, 'N/A')}")

    def _create_edge_index(self) -> None:
        """
        Creates the edge index matrix using sequential indices.

        Requires 'sequential_index' attribute on nodes. Updates `self.edge_index`.
        """
        if self.verbose:
            print("\nCreating edge index matrix...")

        source_indices = []
        target_indices = []

        for source, target in self.graph.edges():
            # Ensure both source and target have been assigned an index
            # (e.g. they weren't unprocessed nodes if any)
            if source in self.node_to_index and target in self.node_to_index:
                source_idx = self.node_to_index[source]
                target_idx = self.node_to_index[target]
                source_indices.append(source_idx)
                target_indices.append(target_idx)

        self.edge_index = [source_indices, target_indices]

        if self.verbose:
            print(f"Edge index created. Total edges in index: {len(source_indices)}")
            print("\nExample edges (source_idx -> target_idx):")
            for i in range(min(5, len(source_indices))):
                s_idx, t_idx = source_indices[i], target_indices[i]
                s_node, t_node = self.index_to_node[s_idx], self.index_to_node[t_idx]
                print(f"  [{s_idx} -> {t_idx}] : ({s_node} -> {t_node})")

    def _prune_by_hierarchy_level(self, max_level: int) -> None:
        """
        Prunes the graph by removing nodes with hierarchy level > max_level.

        Args:
            max_level (int): The maximum hierarchy level to keep (inclusive).
        """
        nodes_before = self.graph.number_of_nodes()
        edges_before = self.graph.number_of_edges()

        nodes_to_remove = [
            node for node, data in self.graph.nodes(data=True)
            if data.get('hierarchy_level', -1) > max_level
        ]

        if self.verbose:
            print(f"\nPruning graph to max hierarchy level {max_level}...")
            print(f"Removing {len(nodes_to_remove)} nodes.")

        self.graph.remove_nodes_from(nodes_to_remove)

        nodes_after = self.graph.number_of_nodes()
        edges_after = self.graph.number_of_edges()

        if self.verbose:
            print(f"Pruning complete.")
            print(f"Nodes: {nodes_before} -> {nodes_after} ({nodes_before - nodes_after} removed)")
            print(f"Edges: {edges_before} -> {edges_after} ({edges_before - edges_after} removed)")

     
        if self.graph.number_of_nodes() > 0:
            self.max_hierarchy_level = max(data.get('hierarchy_level', -1)
                                           for node, data in self.graph.nodes(data=True))
        else:
            self.max_hierarchy_level = -1  # Or None if graph is empty

        # Recalculate sequential indices based on the pruned graph
        self._calculate_sequential_indices()

        if self.verbose:
            print("Sequential indices and edge index recalculated for the pruned graph.")

    def _add_protein_layer_from_gaf(self):
        """
        Add proteins from GAF file as a new layer and create edges between
        proteins and their associated GO terms, using replacement GO terms list.

        Returns:
            self: The updated PathwayNetwork instance (for method chaining)
        """
        import ast

        if self.verbose:
            print("Available columns in GAF DataFrame:")
            print(self.gaf_df.columns.tolist())

        protein_id_col = 'DB_Object_ID'
        protein_symbol_col = 'DB_Object_Symbol'
        replacement_go_id_col = 'Replacement_GO_ID'

        # Get unique proteins
        unique_proteins = self.gaf_df.drop_duplicates(subset=[protein_id_col])

        # Start assigning sequential indices for proteins after the last GO term
        next_index = max(self.node_to_index.values()) + 1
        protein_start_index = next_index

        if self.verbose:
            print(f"\nAdding protein layer starting at index {protein_start_index}")

        # Collect all replacement GO terms associated with each protein
        protein_to_go_terms = {}
        proteins_added = 0
        go_terms_skipped = 0
        total_connections = 0

        for _, row in self.gaf_df.iterrows():
            protein_id = row[protein_id_col]
            protein_symbol = row[protein_symbol_col]
            replacement_go_terms = row[replacement_go_id_col] if isinstance(row[replacement_go_id_col], list) else ast.literal_eval(row[replacement_go_id_col])

            # Skip if no replacement GO terms
            if replacement_go_terms is None:  # this shouldn't happen
                continue

            if protein_id not in protein_to_go_terms:
                protein_to_go_terms[protein_id] = {'symbol': protein_symbol, 'go_terms': set()}

            # Add each replacement GO term to the set
            if isinstance(replacement_go_terms, list):
                for go_id in replacement_go_terms:
                    # Skip if this GO term isn't in the graph 
                    if go_id not in self.graph:
                        go_terms_skipped += 1
                        continue
                    protein_to_go_terms[protein_id]['go_terms'].add(go_id)
            elif replacement_go_terms in self.graph:  # this shouldn't happen
                protein_to_go_terms[protein_id]['go_terms'].add(replacement_go_terms)
            else:  # this shouldn't happen
                go_terms_skipped += 1

        self.added_protein_node_ids = set() 

        # Add proteins to the graph and create edges
        for protein_id, data in protein_to_go_terms.items():
            # Skip proteins with no valid GO terms
            if not data['go_terms']:
                continue

            # Filtering logic
            in_available_nodes = self.available_nodes_in_string is None or protein_id in self.available_nodes_in_string
            in_available_features = self.available_features_in_data is None or protein_id in self.available_features_in_data

            if not (in_available_nodes and in_available_features):
                continue

            # Add the protein to the graph
            protein_node_id = f"PROTEIN:{protein_id}"
            self.graph.add_node(
                protein_node_id,
                name=data['symbol'],
                type='protein',
                hierarchy_level=len(self.layer_indices) - 1,  #  the problem must be here, layer_indices is longer (by one) than the level number
                sequential_index=next_index
            )

            # Update mappings
            self.node_to_index[protein_node_id] = next_index
            self.index_to_node[next_index] = protein_node_id

            # Update id_to_name and name_to_id mappings
            self.id_to_name[protein_node_id] = data['symbol']
            self.name_to_id[data['symbol']] = protein_node_id

            # Add to added_protein_node_ids
            self.added_protein_node_ids.add(protein_id)

            # Create edges between protein and its GO terms
            for go_id in data['go_terms']:
                # Add edge from protein to GO term
                self.graph.add_edge(protein_node_id, go_id, type='annotated_to')
                total_connections += 1

            next_index += 1
            proteins_added += 1

        # Update layer indices to include protein layer
        self.layer_indices.append(next_index)

        if self.verbose:
            print(f"Added {proteins_added} proteins to the graph")
            print(f"Created {total_connections} protein-GO term edges")
            print(f"Skipped {go_terms_skipped} annotations with GO terms not in the graph")

    def _calculate_excluded_proteins(self):
        """
        Calculates which UniProt IDs from the initial input list were not
        successfully added to the graph layer. Populates self.excluded_protein_uniprot_ids.
        """
        if self.verbose:
            print("\nCalculating excluded proteins (UniProt IDs)...")

        # Get the set of intended input UniProt IDs from the original DataFrame
        initial_protein_ids = set(self.protein_input_nodes['gene'].unique())

        # Compare with the set of UniProt IDs actually added to the graph layer
        # Excluded = Initial_Set - Added_Set
        self.excluded_protein_features = initial_protein_ids - self.added_protein_node_ids

        if self.verbose:
            print(f"Initial intended UniProt IDs list size: {len(initial_protein_ids)}")
            print(f"UniProt IDs added to graph layer: {len(self.added_protein_node_ids)}")
            print(f"UniProt IDs excluded (intended but not added): {len(self.excluded_protein_features)}")
            if self.excluded_protein_features and len(self.excluded_protein_features) < 20: # Print a few examples if not too many
                 print(f"Examples of excluded UniProt IDs: {list(self.excluded_protein_features)[:10]}")
            elif not self.excluded_protein_features:
                 print("No intended input proteins were excluded.")

    def _prune_unreachable_nodes(self):
        """
        Remove nodes from which no protein node can be reached.
        This filters the graph to only include nodes in pathways used by proteins.

        Returns:
            self: The updated PathwayNetwork instance (for method chaining)
        """
        from collections import deque, defaultdict

        # Find max hierarchy level (proteins are at the highest level)
        max_level = max(nx.get_node_attributes(self.graph, 'hierarchy_level').values())

        # Find all protein nodes (nodes at the highest hierarchy level)
        protein_nodes = [node for node, data in self.graph.nodes(data=True)
                         if data.get('hierarchy_level') == max_level]

        if self.verbose:
            print(f"Found {len(protein_nodes)} protein nodes at hierarchy level {max_level}")
            print("Starting forward traversal from protein nodes...")

        # Identify nodes that are reachable FROM protein nodes using forward BFS
        reachable_nodes = set()

        # Use a single BFS from all protein nodes at once
        queue = deque(protein_nodes)
        reachable_nodes.update(protein_nodes)  # Proteins can reach themselves

        while queue:
            current = queue.popleft()

            # Check all neighbors in the forward direction (nodes that current points to)
            for neighbor in self.graph.neighbors(current):
                if neighbor not in reachable_nodes:
                    reachable_nodes.add(neighbor)
                    queue.append(neighbor)

        # Identify nodes to remove
        nodes_to_remove = [node for node in self.graph.nodes() if node not in reachable_nodes]

        if self.verbose:
            # Count nodes by hierarchy level before pruning
            level_counts_before = defaultdict(int)
            level_counts_removed = defaultdict(int)
            level_counts_after = defaultdict(int)

            for node in self.graph.nodes():
                level = self.graph.nodes[node].get('hierarchy_level', -1)
                level_counts_before[level] += 1

                if node in reachable_nodes:
                    level_counts_after[level] += 1
                else:
                    level_counts_removed[level] += 1

            nodes_before = self.graph.number_of_nodes()
            edges_before = self.graph.number_of_edges()

        # Remove unreachable nodes from the graph
        self.graph.remove_nodes_from(nodes_to_remove)

        if self.verbose:
            nodes_after = self.graph.number_of_nodes()
            edges_after = self.graph.number_of_edges()

            print(f"Original graph: {nodes_before} nodes, {edges_before} edges")
            print(f"Pruned graph: {nodes_after} nodes, {edges_after} edges")
            print(
                f"Removed {len(nodes_to_remove)} unreachable nodes ({len(nodes_to_remove) / nodes_before:.1%} of total)")

            # Print nodes removed by level
            print("\nNodes by hierarchy level:")
            print(f"{'Level':<8} {'Before':<8} {'Removed':<8} {'After':<8} {'% Removed':<10}")
            print("-" * 45)

            for level in sorted(level_counts_before.keys()):
                level_name = "Protein" if level == max_level else f"Level {level}"
                before = level_counts_before[level]
                removed = level_counts_removed[level]
                after = level_counts_after[level]
                percent = removed / before * 100 if before > 0 else 0

                print(f"{level_name:<8} {before:<8} {removed:<8} {after:<8} {percent:.1f}%")

        # Recalculate sequential indices and update mappings
        self.max_hierarchy_level = max_level
        self._calculate_sequential_indices()

    def _preproc_gaf(self) -> pd.DataFrame:
        """
        Preprocess the GAF DataFrame to ensure it contains the necessary columns
        and formats for adding a protein layer.

        Returns:
            gaf_df: The preprocessed GAF DataFrame.
        """
        # Read GAF file
        self.gaf_df = read_gaf(self.gaf_file_path)

        if self.verbose:
            print("\nPreprocessing GAF DataFrame...")
            print("Filtering using qualifiers...")
            print("Length of GAF DataFrame before filtering:", len(self.gaf_df))

        # Filter by qualifiers
        qualifiers_to_include = [
            'part_of',
            'located_in',
            'is_active_in',
            'involved_in',
            'enables',
            'contributes_to',
            'colocalizes_with',
            'acts_upstream_of_positive_effect',
            'acts_upstream_of_or_within_positive_effect'
        ]

        qualifiers = unique(self.gaf_df['Qualifier'])
        self.gaf_df = self.gaf_df[self.gaf_df['Qualifier'].isin(qualifiers_to_include)]

        if self.verbose:
            print("Length of GAF DataFrame after filtering:", len(self.gaf_df))
            print("Qualifiers in GAF DataFrame:", qualifiers_to_include)

        # Add replacement GO terms
        if self.verbose:
            print("Adding replacement GO terms...")
        self.gaf_df = add_replacement_go_terms(self.gaf_df, self.graph, self.max_level)

        # Subset the GAF DataFrame (based on the protein_input_nodes) to only include relevant proteins
        if self.verbose:
            print("Filtering GAF DataFrame to include only relevant proteins...")
        self.gaf_df = self.gaf_df[self.gaf_df['DB_Object_ID'].isin(self.protein_input_nodes['gene'])]

        # Write the updated GAF DataFrame to a new file
        if self.verbose:
            print("Writing preprocessed GAF DataFrame to file...")
        output_gaf_path = self.gaf_file_path.replace('.gaf', '_processed_filtered_8.gaf')
        write_gaf_with_replacements(self.gaf_df, output_gaf_path)

        return self.gaf_df

    def _remove_first_layer(self):
        """
        Remove the first layer of the graph (level 0) and update the hierarchy levels.
        """
        from collections import defaultdict
        if self.verbose:
            print("\nRemoving first layer (level 0) from the graph...")

        # Find all nodes at level 0
        level_0_nodes = [node for node, data in self.graph.nodes(data=True)
                         if data.get('hierarchy_level') == 0]
        
        # Remove level 0 nodes from the graph
        self.graph.remove_nodes_from(level_0_nodes)
        
        # Check if there were any protein nodes only connected to the first layer
        # protein nodes are on the last layer
        protein_nodes = [node for node, data in self.graph.nodes(data=True)
                         if data.get('hierarchy_level') == self.max_hierarchy_level]
        protein_nodes_connected_to_level_0 = [node for node in protein_nodes if self.graph.out_degree(node) == 0]

        if protein_nodes_connected_to_level_0:

            if self.verbose:
                nodes_before = self.graph.number_of_nodes()
                edges_before = self.graph.number_of_edges()

            # Remove them from the graph
            self.graph.remove_nodes_from(protein_nodes_connected_to_level_0)

            # Update self.added_protein_node_ids and self.excluded_protein_features
            protein_nodeids_connected_to_level_0 = [node.replace("PROTEIN:", "") for node in protein_nodes_connected_to_level_0]
            self.added_protein_node_ids = set(self.added_protein_node_ids) - set(protein_nodeids_connected_to_level_0)
            self.excluded_protein_features.update(set(protein_nodeids_connected_to_level_0))


            if self.verbose:
                print(f"Removed {len(protein_nodes_connected_to_level_0)} protein nodes connected only to level 0.")
                print(f"Number of protein nodes after removal: {len(protein_nodes) - len(protein_nodes_connected_to_level_0)}")

                nodes_after = self.graph.number_of_nodes()
                edges_after = self.graph.number_of_edges()

                print(f"Original graph: {nodes_before} nodes, {edges_before} edges")
                print(f"Pruned graph: {nodes_after} nodes, {edges_after} edges")

                

                
        # Update hierarchy levels for remaining nodes
        for node in self.graph.nodes():
            current_level = self.graph.nodes[node].get('hierarchy_level', -1)
            if current_level > 0:
                self.graph.nodes[node]['hierarchy_level'] = current_level - 1

        self.max_hierarchy_level -= 1
        if self.verbose:
            # Print nodes by hierarchy level after removal
            level_counts_after = defaultdict(int)
            for node in self.graph.nodes():
                level = self.graph.nodes[node].get('hierarchy_level', -1)
                level_counts_after[level] += 1
            print("\nNodes by hierarchy level after removal:")
            print(f"{'Level':<8} {'After':<8}")
            print("-" * 20)
            for level in sorted(level_counts_after.keys()):
                level_name = "Protein" if level == self.max_hierarchy_level else f"Level {level}"
                after = level_counts_after[level]
                print(f"{level_name:<8} {after:<8}")

        # Recalculate sequential indices
        self._calculate_sequential_indices()

        if self.verbose:
            print(f"Removed {len(level_0_nodes)} nodes from level 0.")
            print(f"New max hierarchy level: {self.max_hierarchy_level}")
            print(f"Sequential indices recalculated.")

    def _build_network(self):
        """
        Build the network by calculating hierarchy levels, sequential indices,
        and creating the edge index.
        """

        # Prune the graph to include only specified root nodes
        self._prune_using_root_nodes(self.root_nodes_to_include)

        # Calculate hierarchy levels
        self._calculate_hierarchy_levels()

        # Add sequential indices
        self._calculate_sequential_indices()

        # Process the GAF file if provided
        if not self.gaf_is_preprocessed:
            self._preproc_gaf()
        else:
            self.gaf_df = read_preprocessed_gaf(self.preprocessed_gaf_file_path)

        # Prune by hierarchy level if specified
        if self.max_hierarchy_level is not None:
            self._prune_by_hierarchy_level(self.max_level)

        # Add protein layer from GAF file
        self._add_protein_layer_from_gaf()

        # Calculate excluded proteins
        self._calculate_excluded_proteins()

        # Prune unreachable nodes
        self._prune_unreachable_nodes()

        # Remove the first layer (level 0) 
        if self.max_hierarchy_level > 0:
            self._remove_first_layer()

        # Create edge index
        self._create_edge_index()


