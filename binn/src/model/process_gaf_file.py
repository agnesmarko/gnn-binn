import pandas as pd
import gzip
from collections import deque


def check_gaf_version(file_path):
    """
    Check the GAF version from file header.

    Parameters:
    -----------
    file_path : str
        Path to the GAF file. Can be gzipped (.gz) or plain text.

    Returns:
    --------
    str
        GAF version if found, otherwise "Unknown"
    """
    # Open file (handling gzipped files if needed)
    if file_path.endswith('.gz'):
        import gzip
        opener = gzip.open(file_path, 'rt')
    else:
        opener = open(file_path, 'r')

    try:
        with opener as f:
            # Check the first few lines for version information
            for i, line in enumerate(f):
                if i > 20:  # Only check first 20 lines
                    break

                # Look for format version in header comments
                if line.startswith('!gaf-version:'):
                    return line.strip().split('!gaf-version:')[1].strip()
                elif '!GAF-VERSION' in line:
                    return line.split('!GAF-VERSION')[1].strip()

        return "Unknown - No version header found"

    except Exception as e:
        return f"Error: {str(e)}"



def read_gaf(file_path):
    """
    Read a Gene Association File (GAF) into a pandas DataFrame.

    Parameters:
    -----------
    file_path : str
        Path to the GAF file. Can be gzipped (.gz) or plain text.

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the GAF data with proper column names.
    """
    # GAF 2.2 format column names
    column_names = [
        'DB',
        'DB_Object_ID',
        'DB_Object_Symbol',
        'Qualifier',
        'GO_ID',
        'DB_Reference',
        'Evidence_Code',
        'With_From',
        'Aspect',
        'DB_Object_Name',
        'DB_Object_Synonym',
        'DB_Object_Type',
        'Taxon',
        'Date',
        'Assigned_By',
        'Annotation_Extension',
        'Gene_Product_Form_ID'
    ]

    # Check if file is gzipped
    if file_path.endswith('.gz'):
        # Open gzipped file
        with gzip.open(file_path, 'rt') as f:
            # Skip comment lines that start with '!'
            df = pd.read_csv(f, sep='\t', comment='!', names=column_names, low_memory=False)
    else:
        # Open regular file
        df = pd.read_csv(file_path, sep='\t', comment='!', names=column_names, low_memory=False)

    return df


def read_preprocessed_gaf(file_path):
    """
    Read a preprocessed Gene Association File (GAF) that includes a Replacement_GO_ID column.

    Parameters:
    -----------
    file_path : str
        Path to the preprocessed GAF file. Can be gzipped (.gz) or plain text.

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the GAF data including the Replacement_GO_ID column.
    """
    import pandas as pd
    import gzip

    # GAF 2.2 format column names plus the Replacement_GO_ID column
    column_names = [
        'DB',
        'DB_Object_ID',
        'DB_Object_Symbol',
        'Qualifier',
        'GO_ID',
        'DB_Reference',
        'Evidence_Code',
        'With_From',
        'Aspect',
        'DB_Object_Name',
        'DB_Object_Synonym',
        'DB_Object_Type',
        'Taxon',
        'Date',
        'Assigned_By',
        'Annotation_Extension',
        'Gene_Product_Form_ID',
        'Replacement_GO_ID'  # Added column for replacement GO terms
    ]

    # Check if file is gzipped
    if file_path.endswith('.gz'):
        with gzip.open(file_path, 'rt') as f:
            df = pd.read_csv(f, sep='\t', comment='!', names=column_names, low_memory=False)
    else:
        df = pd.read_csv(file_path, sep='\t', comment='!', names=column_names, low_memory=False)

    return df


def add_gene_ids(gaf_df, gene2uniprot_path):
    """Alternative approach using pandas merge"""
    import pandas as pd

    # Read the gene2uniprot mapping file
    gene_uniprot_df = pd.read_csv(gene2uniprot_path, sep='\t', header=None,
                                 names=['Gene_ID', 'Gene_Symbol', 'UniProt_ID'])

    # Clean the data
    gene_uniprot_df = gene_uniprot_df.dropna(subset=['UniProt_ID'])

    # Filter GAF DataFrame for UniProtKB entries
    uniprot_mask = gaf_df['DB'] == 'UniProtKB'

    # Merge the DataFrames
    result = gaf_df.copy()
    result.loc[uniprot_mask, 'Gene_ID'] = pd.merge(
        gaf_df[uniprot_mask],
        gene_uniprot_df[['UniProt_ID', 'Gene_ID']],
        left_on='DB_Object_ID',
        right_on='UniProt_ID',
        how='left'
    )['Gene_ID']

    return result


def add_replacement_go_terms(gaf_df, graph, max_level, go_id_col='GO_ID'):
    """
    Add a column to the GAF dataframe with replacement GO terms for those that would be pruned.

    For each GO term in the GAF file with hierarchy_level > max_level:
    - Find its closest ancestors with hierarchy_level ≤ max_level
    - These are "frontier" nodes - the first valid ancestors encountered in each path up the hierarchy

    Args:
        gaf_df: Pandas DataFrame containing GAF annotation data
        graph: The original GO graph with hierarchy_level attribute
        max_level: Maximum hierarchy level for pruning
        go_id_col: Column name in gaf_df containing GO term IDs

    Returns:
        updated_gaf_df: GAF DataFrame with new column 'Replacement_GO_ID'
    """

    # Make a copy of the input dataframe
    updated_gaf_df = gaf_df.copy()

    # Create a cache to avoid recalculating replacements for the same GO term
    replacement_cache = {}

    # Function to find closest ancestor GO terms with hierarchy_level ≤ max_level
    def find_frontier_ancestors(go_term):
        # If we've already calculated replacements for this term, use the cached result
        if go_term in replacement_cache:
            return replacement_cache[go_term]

        # If the term isn't in the graph, return None
        if go_term not in graph:
            replacement_cache[go_term] = None
            return None

        # Find frontier ancestors with BFS
        frontier_ancestors = set()
        visited = set()
        queue = deque([go_term])

        while queue:
            current_term = queue.popleft()

            # Skip if we've visited this term before
            if current_term in visited:
                continue

            visited.add(current_term)

            # Check immediate parents
            has_valid_parents = False

            for parent in graph.successors(current_term):
                parent_level = graph.nodes[parent]['hierarchy_level']

                # If this parent has acceptable level, add it to frontier_ancestors
                if parent_level <= max_level:
                    frontier_ancestors.add(parent)
                    has_valid_parents = True
                    # Don't explore beyond this parent - it's part of the frontier
                else:
                    # If parent level is still too high, add to queue to explore further
                    queue.append(parent)

            # If no valid parents were found from this node, keep exploring upward
            if not has_valid_parents and current_term != go_term:
                continue

        # Convert set to list for consistent order
        frontier_list = sorted(list(frontier_ancestors))

        # Cache the result
        replacement_cache[go_term] = frontier_list

        return frontier_list

    # Count statistics
    total_annotations = len(updated_gaf_df)
    terms_needing_replacement = 0
    terms_with_replacements = 0
    terms_without_replacements = 0
    terms_not_in_graph = 0

    # Process each row in the dataframe
    replacement_go_ids = []

    for _, row in updated_gaf_df.iterrows():
        go_term = row[go_id_col]

        # Skip if term isn't in the graph
        if go_term not in graph:
            replacement_go_ids.append(None)
            terms_not_in_graph += 1
            continue

        # Check if this term needs replacement
        term_level = graph.nodes[go_term]['hierarchy_level']
        if term_level <= max_level:
            # No replacement needed - keep the original term
            replacement_go_ids.append([go_term])  # Store as a list with one item
        else:
            # Term needs replacement
            terms_needing_replacement += 1

            # Find frontier ancestors
            replacements = find_frontier_ancestors(go_term)

            if replacements and len(replacements) > 0:
                terms_with_replacements += 1
                replacement_go_ids.append(replacements)  # Store the list directly
            else:
                terms_without_replacements += 1
                replacement_go_ids.append(None)

    # Add the replacement column to the dataframe
    updated_gaf_df['Replacement_GO_ID'] = replacement_go_ids

    # Print statistics
    print(f"Processed {total_annotations} GO term annotations")
    print(f"Terms not in graph: {terms_not_in_graph}")
    print(
        f"Terms needing replacement: {terms_needing_replacement} ({terms_needing_replacement / total_annotations:.1%})")
    print(f"Terms with replacements found: {terms_with_replacements}")
    print(f"Terms without replacements: {terms_without_replacements}")

    return updated_gaf_df



def write_gaf_with_replacements(dataframe, output_path, include_header=True):
    """
    Write a DataFrame to a GAF file format, preserving original annotations
    and adding new rows for replacement GO terms.

    Args:
        dataframe: Pandas DataFrame containing GAF data with Replacement_GO_ID column
        output_path: Path where the GAF file will be saved
        include_header: Whether to include GAF format header/comments (Default: True)
    """
    import pandas as pd

    # Standard GAF column names
    gaf_columns = [
        'DB', 'DB_Object_ID', 'DB_Object_Symbol', 'Qualifier', 'GO_ID',
        'DB_Reference', 'Evidence_Code', 'With_From', 'Aspect',
        'DB_Object_Name', 'DB_Object_Synonym', 'DB_Object_Type',
        'Taxon', 'Date', 'Assigned_By', 'Annotation_Extension',
        'Gene_Product_Form_ID', 'Replacement_GO_ID'
    ]

    # Open file for writing
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write standard GAF header if requested
        if include_header:
            f.write("!gaf-version: 2.2\n")
            f.write("!generated-by: Python script with GO term replacements\n")
            f.write(f"!date-generated: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n")
            f.write("!\n")

        # First, write all original rows
        for _, row in dataframe.iterrows():
            # Format and write the original annotation
            values = [str(row[col]).replace('\t', ' ').replace('\n', ' ') for col in gaf_columns]
            f.write('\t'.join(values) + '\n')

            # Then add rows for each replacement GO term (if any exist)
            replacements = row.get('Replacement_GO_ID')
            if replacements is not None and len(replacements) > 0 and (
                # Only add replacements if they're different from the original
                len(replacements) > 1 or replacements[0] != row['GO_ID']
            ):
                for replacement in replacements:
                    # Skip if replacement is the same as original GO ID
                    if replacement == row['GO_ID']:
                        continue

                    # Create a modified row for each replacement
                    new_row = row.copy()
                    # Replace the GO_ID with the replacement
                    new_row['GO_ID'] = replacement

                    # Add a note to the Qualifier field indicating this is a replacement
                    qualifier = new_row['Qualifier']
                    if qualifier and qualifier != '':
                        new_row['Qualifier'] = qualifier + '|replaced_from:' + row['GO_ID']
                    else:
                        new_row['Qualifier'] = 'replaced_from:' + row['GO_ID']

                    # Format and write the replacement row
                    values = [str(new_row[col]).replace('\t', ' ').replace('\n', ' ') for col in gaf_columns]
                    f.write('\t'.join(values) + '\n')

    print(f"GAF file with original annotations and replacements successfully written to {output_path}")

