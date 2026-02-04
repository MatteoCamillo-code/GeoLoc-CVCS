"""Data loader utility for handling multiple prediction CSV formats."""

import pandas as pd
from pathlib import Path
from typing import Union, Optional


def load_predictions_csv(
    csv_path: Union[str, Path],
    image_folder: Optional[Union[str, Path]] = None,
    normalize_ids: bool = True,
) -> pd.DataFrame:
    """
    Load predictions CSV and normalize columns.
    
    Handles two formats:
    1. train_val_with_predictions.csv and test_with_predictions.csv
    2. mp16_with_predictions.csv
    
    Returns DataFrame with columns: id, latitude, longitude, predicted_label
    
    Args:
        csv_path: Path to the CSV file
        image_folder: Optional path to the image folder
        normalize_ids: If True, clean up IDs (remove .jpg extensions)
        
    Returns:
        DataFrame with normalized columns
    """
    df = pd.read_csv(csv_path)
    
    # Select only the columns we need
    required_cols = ['id', 'latitude', 'longitude', 'predicted_label']
    #if df has column region, keep it
    if 'region' in df.columns:
        required_cols.append('region')
    df = df[required_cols].copy()
    
    # Normalize IDs - remove .jpg extension if present
    if normalize_ids:
        df['id'] = df['id'].astype(str).str.replace('.jpg$', '', regex=True)
        # add folder_name/ before id if image_folder is provided not the full absolute path
        if image_folder is not None:
            if 'region' in df.columns:
                # TODO: this works only for osv5m folder structure
                df['id'] = "train_images/" + df['region'].astype(str) + '/' + (df['id'])
                required_cols.remove('region')
            df['id'] = str(image_folder) + '/' + (df['id'])       
    
    return df[required_cols]


def load_multiple_predictions(
    csv_paths: list,
    combine: bool = True,
    normalize_ids: bool = True,
) -> Union[list, pd.DataFrame]:
    """
    Load multiple prediction CSVs.
    
    Args:
        csv_paths: List of paths to CSV files
        combine: If True, combine all DataFrames into one; else return list
        normalize_ids: If True, clean up IDs
        
    Returns:
        Combined DataFrame or list of DataFrames
    """
    dfs = []
    for path in csv_paths:
        df = load_predictions_csv(path, normalize_ids=normalize_ids)
        dfs.append(df)
    
    if combine:
        # Remove duplicates (keep first occurrence)
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['id'], keep='first')
        return combined_df
    else:
        return dfs


def get_metadata(
    project_root: Path,
    include_mp16: bool = True,
    include_osv_mini: bool = True,
) -> pd.DataFrame:
    """
    Load prediction metadata from available CSV files.
    
    Args:
        project_root: Path to project root
        include_mp16: Include mp16_with_predictions.csv if available
        include_osv_mini: Include osv_mini_with_predictions.csv if available
        
    Returns:
        Combined DataFrame with id, latitude, longitude, predicted_label
    """
    dfs = []
    
    if include_osv_mini:
        osv_mini_path = project_root / "data" / "metadata" / "places-classification" / "osv_mini_with_predictions.csv"
        if osv_mini_path.exists():
            df = load_predictions_csv(osv_mini_path, image_folder="osv5m")
            dfs.append(df)
            print(f"Loaded {len(df)} records from osv_mini_with_predictions.csv")
    
    if include_mp16:
        mp16_path = project_root / "data" / "metadata" / "places-classification" / "mp16_with_predictions.csv"
        if mp16_path.exists():
            df = load_predictions_csv(mp16_path, image_folder="mp16_images")
            dfs.append(df)
            print(f"Loaded {len(df)} records from mp16_with_predictions.csv")
    
    if not dfs:
        raise FileNotFoundError(
            "No prediction CSV files found. "
            "Please ensure osv_mini_with_predictions.csv or mp16_with_predictions.csv exist."
        )
    
    # Combine and remove duplicates
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=['id'], keep='first')
    
    print(f"Total unique records after combining: {len(combined)}")
    print(f"Label distribution:\n{combined['predicted_label'].value_counts()}")
    
    return combined
