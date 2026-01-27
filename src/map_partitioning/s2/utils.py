import s2sphere
import numpy as np
import pandas as pd

def get_s2_cell_id(lat, lng, level=30):
    """
    Returns the S2 cell ID for a given latitude and longitude at a specific level.
    """
    p = s2sphere.LatLng.from_degrees(lat, lng)
    return s2sphere.CellId.from_lat_lng(p).parent(level)

def get_cell_vertices(cell_id):
    """
    Returns the four vertices of an S2 cell as (longitude, latitude) tuples.
    Used for plotting cell boundaries.
    """
    cell = s2sphere.Cell(cell_id)
    vertices = []
    for i in range(4):
        vertex = s2sphere.LatLng.from_point(cell.get_vertex(i))
        vertices.append((vertex.lng().degrees, vertex.lat().degrees))
    return vertices

def fast_assign_label(point_s2_id, lookup_dict):
    """
    Assigns a label to a point by finding its parent cell in the lookup dictionary.
    Checks from the most specific level (30) up to the root.
    """
    for lvl in range(30, -1, -1):
        parent_token = point_s2_id.parent(lvl).to_token()
        if parent_token in lookup_dict:
            return parent_token
    return '-1'

def perform_train_val_split(df, split_ratio=0.85, random_seed=42):
    """
    Splits the dataframe into training and validation sets.
    """
    np.random.seed(random_seed)
    train_len = int(split_ratio * len(df))
    train_indices = np.sort(np.random.choice(len(df), size=train_len, replace=False))
    
    df['is_train'] = 0
    df.loc[train_indices, 'is_train'] = 1
    
    df_train = df[df['is_train'] == 1].copy()
    df_val = df[df['is_train'] == 0].copy()
    return df_train, df_val