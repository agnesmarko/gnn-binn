import pandas as pd
import gzip
from collections import deque


def check_gaf_version(file_path):
    # check the gaf version from file header.

    # open file (handling gzipped files if needed)
    if file_path.endswith('.gz'):
        import gzip
        opener = gzip.open(file_path, 'rt')
    else:
        opener = open(file_path, 'r')

    try:
        with opener as f:
            # check the first few lines for version information
            for i, line in enumerate(f):
                if i > 20:  # only check first 20 lines
                    break

                # look for format version in header comments
                if line.startswith('!gaf-version:'):
                    return line.strip().split('!gaf-version:')[1].strip()
                elif '!GAF-VERSION' in line:
                    return line.split('!GAF-VERSION')[1].strip()

        return "unknown - no version header found"

    except Exception as e:
        return f"error: {str(e)}"


def read_gaf(file_path):
    # read a gene association file (gaf) into a pandas dataframe.

    # gaf 2.2 format column names
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

    # check if file is gzipped
    if file_path.endswith('.gz'):
        # open gzipped file
        with gzip.open(file_path, 'rt') as f:
            # skip comment lines that start with '!'
            df = pd.read_csv(f, sep='\t', comment='!', names=column_names, low_memory=False)
    else:
        # open regular file
        df = pd.read_csv(file_path, sep='\t', comment='!', names=column_names, low_memory=False)

    return df


def read_preprocessed_gaf(file_path):
    # read a preprocessed gene association file (gaf) that includes a replacement_go_id column.

    import pandas as pd
    import gzip

    # gaf 2.2 format column names plus the replacement_go_id column
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
        'Replacement_GO_ID'  # added column for replacement go terms
    ]

    # check if file is gzipped
    if file_path.endswith('.gz'):
        with gzip.open(file_path, 'rt') as f:
            df = pd.read_csv(f, sep='\t', comment='!', names=column_names, low_memory=False)
    else:
        df = pd.read_csv(file_path, sep='\t', comment='!', names=column_names, low_memory=False)

    return df


def add_replacement_go_terms(gaf_df, graph, max_level, go_id_col='GO_ID'):
    # add a column to the gaf dataframe with replacement go terms for those that would be pruned.
    # for each go term in the gaf file with hierarchy_level > max_level:
    # - find its closest ancestors with hierarchy_level ≤ max_level
    # - these are "frontier" nodes - the first valid ancestors encountered in each path up the hierarchy

    # make a copy of the input dataframe
    updated_gaf_df = gaf_df.copy()

    # create a cache to avoid recalculating replacements for the same go term
    replacement_cache = {}

    # function to find closest ancestor go terms with hierarchy_level ≤ max_level
    def find_frontier_ancestors(go_term):
        # if we've already calculated replacements for this term, use the cached result
        if go_term in replacement_cache:
            return replacement_cache[go_term]

        # if the term isn't in the graph, return none
        if go_term not in graph:
            replacement_cache[go_term] = None
            return None

        # find frontier ancestors with bfs
        frontier_ancestors = set()
        visited = set()
        queue = deque([go_term])

        while queue:
            current_term = queue.popleft()

            # skip if we've visited this term before
            if current_term in visited:
                continue

            visited.add(current_term)

            # check immediate parents
            has_valid_parents = False

            for parent in graph.successors(current_term):
                parent_level = graph.nodes[parent]['hierarchy_level']

                # if this parent has acceptable level, add it to frontier_ancestors
                if parent_level <= max_level:
                    frontier_ancestors.add(parent)
                    has_valid_parents = True
                    # don't explore beyond this parent - it's part of the frontier
                else:
                    # if parent level is still too high, add to queue to explore further
                    queue.append(parent)

            # if no valid parents were found from this node, keep exploring upward
            if not has_valid_parents and current_term != go_term:
                continue

        # convert set to list for consistent order
        frontier_list = sorted(list(frontier_ancestors))

        # cache the result
        replacement_cache[go_term] = frontier_list

        return frontier_list

    # count statistics
    total_annotations = len(updated_gaf_df)
    terms_needing_replacement = 0
    terms_with_replacements = 0
    terms_without_replacements = 0
    terms_not_in_graph = 0

    # process each row in the dataframe
    replacement_go_ids = []

    for _, row in updated_gaf_df.iterrows():
        go_term = row[go_id_col]

        # skip if term isn't in the graph
        if go_term not in graph:
            replacement_go_ids.append(None)
            terms_not_in_graph += 1
            continue

        # check if this term needs replacement
        term_level = graph.nodes[go_term]['hierarchy_level']
        if term_level <= max_level:
            # no replacement needed - keep the original term
            replacement_go_ids.append([go_term])  # store as a list with one item
        else:
            # term needs replacement
            terms_needing_replacement += 1

            # find frontier ancestors
            replacements = find_frontier_ancestors(go_term)

            if replacements and len(replacements) > 0:
                terms_with_replacements += 1
                replacement_go_ids.append(replacements)  # store the list directly
            else:
                terms_without_replacements += 1
                replacement_go_ids.append(None)

    # add the replacement column to the dataframe
    updated_gaf_df['Replacement_GO_ID'] = replacement_go_ids

    # print statistics
    print(f"processed {total_annotations} go term annotations")
    print(f"terms not in graph: {terms_not_in_graph}")
    print(
        f"terms needing replacement: {terms_needing_replacement} ({terms_needing_replacement / total_annotations:.1%})")
    print(f"terms with replacements found: {terms_with_replacements}")
    print(f"terms without replacements: {terms_without_replacements}")

    return updated_gaf_df



def write_gaf_with_replacements(dataframe, output_path, include_header=True):
    # write a dataframe to a gaf file format, preserving original annotations
    # and adding new rows for replacement go terms.

    import pandas as pd

    # standard gaf column names
    gaf_columns = [
        'DB', 'DB_Object_ID', 'DB_Object_Symbol', 'Qualifier', 'GO_ID',
        'DB_Reference', 'Evidence_Code', 'With_From', 'Aspect',
        'DB_Object_Name', 'DB_Object_Synonym', 'DB_Object_Type',
        'Taxon', 'Date', 'Assigned_By', 'Annotation_Extension',
        'Gene_Product_Form_ID', 'Replacement_GO_ID'
    ]

    # open file for writing
    with open(output_path, 'w', encoding='utf-8') as f:
        # write standard gaf header if requested
        if include_header:
            f.write("!gaf-version: 2.2\n")
            f.write("!generated-by: python script with go term replacements\n")
            f.write(f"!date-generated: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n")
            f.write("!\n")

        # first, write all original rows
        for _, row in dataframe.iterrows():
            # format and write the original annotation
            values = [str(row[col]).replace('\t', ' ').replace('\n', ' ') for col in gaf_columns]
            f.write('\t'.join(values) + '\n')

            # then add rows for each replacement go term (if any exist)
            replacements = row.get('Replacement_GO_ID')
            if replacements is not None and len(replacements) > 0 and (
                # only add replacements if they're different from the original
                len(replacements) > 1 or replacements[0] != row['GO_ID']
            ):
                for replacement in replacements:
                    # skip if replacement is the same as original go id
                    if replacement == row['GO_ID']:
                        continue

                    # create a modified row for each replacement
                    new_row = row.copy()
                    # replace the go_id with the replacement
                    new_row['GO_ID'] = replacement

                    # add a note to the qualifier field indicating this is a replacement
                    qualifier = new_row['Qualifier']
                    if qualifier and qualifier != '':
                        new_row['Qualifier'] = qualifier + '|replaced_from:' + row['GO_ID']
                    else:
                        new_row['Qualifier'] = 'replaced_from:' + row['GO_ID']

                    # format and write the replacement row
                    values = [str(new_row[col]).replace('\t', ' ').replace('\n', ' ') for col in gaf_columns]
                    f.write('\t'.join(values) + '\n')

    print(f"gaf file with original annotations and replacements successfully written to {output_path}")