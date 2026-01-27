import pandas as pd
from tqdm import tqdm
from src.map_partitioning.s2.utils import get_cell_vertices, fast_assign_label

def partition(cell_id, points, tau_max, leaf_cells_result):
    """
    Internal recursive function to perform the partitioning.
    """
    count = len(points)
    # Base case: if count is within threshold or maximum level reached
    if count <= tau_max or cell_id.level() >= 30:
        if count > 0:
            leaf_cells_result.append({'cell_id': cell_id, 'count': count})
        return

    # Recursive step: subdivide into 4 children
    for i in range(4):
        child = cell_id.child(i)
        # Filter points that belong to the current child cell
        child_points = [p for p in points if child.contains(p)]
        if child_points:
            partition(child, child_points, tau_max, leaf_cells_result)

def run_partitioning(df_points, tau_max, start_level=4):
    """
    Recursively partitions geographical points into S2 cells based on a maximum
    threshold (tau_max). Points are grouped into cells; if a cell's point count
    exceeds tau_max, it's subdivided into child cells until the condition is met
    or a maximum S2 cell level is reached.

    Args:
        df_points (pd.DataFrame): DataFrame containing finer level S2 CellIds of
        points in a column named 's2_cell'.
        tau_max (int): The maximum number of points allowed in a leaf cell.
        start_level (int, optional): The initial S2 cell level to start partitioning from.
        Defaults to 4 (a relatively coarse level).

    Returns:
        list: A list of dictionaries, where each dictionary represents a leaf cell
        and contains its 'cell_id' (s2sphere.CellId) and 'count' of points.
    """
    leaf_cells_result = []
    # Get unique initial parent cells at start_level to begin partitioning
    initial_cells = df_points['s2_cell'].apply(lambda x: x.parent(start_level)).unique()

    # Process each initial cell separately
    for start_id in initial_cells:
        # Filter all points relevant to this initial cell
        relevant_points = [p for p in df_points['s2_cell'] if start_id.contains(p)]
        # Start recursive partitioning for this set of points using the external function
        partition(start_id, relevant_points, tau_max, leaf_cells_result)
        
    return leaf_cells_result

def s2_partitioning(df_train: pd.DataFrame, configs: list):
    """
    Applies S2 cell partitioning to the training DataFrame based on different configurations.
    For each configuration, it identifies leaf cells, creates lookup dictionaries, and assigns
    corresponding S2 cell labels to the training data.
    """
    trained_partitions = {} # Stores lists of leaf_cells for plotting
    trained_lookups = {} # Stores lookup dictionaries for fast matching

    print("\nStarting S2 Cell Partitioning and Labeling")

    for cfg in configs:
        print(f"\nPartitioning {cfg['name']} (tau_max={cfg['tau_max']})...")

        # Run partitioning to get leaf cells for the current configuration
        leaves = run_partitioning(df_train, cfg['tau_max'])
        trained_partitions[cfg['name']] = leaves

        # Print the number of cells created for this configuration
        num_cells = len(leaves)
        print(f"Number of cells (classes) created: {num_cells}")

        # Create a lookup dictionary: maps cell token to cell token (used as label)
        lookup = {leaf['cell_id'].to_token(): leaf['cell_id'].to_token() for leaf in leaves}
        trained_lookups[cfg['name']] = lookup

        # Assign labels to the training DataFrame using the fast_assign_label function
        col_name = f"label_{cfg['name']}"
        df_train.loc[:, col_name] = [fast_assign_label(p, lookup) for p in tqdm(df_train['s2_cell'], desc=f" Mapping {cfg['name']}")]

    print("\n \nS2 Cell Partitioning and Labeling completed.")
    return trained_partitions, trained_lookups, df_train