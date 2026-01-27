import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib.patches as patches
from tqdm import tqdm

# Note the absolute import here
from src.map_partitioning.s2.utils import get_cell_vertices, fast_assign_label

def plot_density_map(df: pd.DataFrame, title: str, x_label: str, y_label: str, 
                     gridsize: int = 60, figsize: tuple = (12, 8), 
                     x_lim: tuple = (-126, -66), y_lim: tuple = (24, 50)):
    """
    Generates a hexbin density plot of geographical points on a USA map.
    This helps visualize the distribution of data points before and after partitioning.
    """
    print("Loading USA map for density plot...")
    # Load geographic boundaries for background
    usa_url = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
    usa = gpd.read_file(usa_url)
    
    # Filter to focus on the contiguous United States
    usa = usa[~usa['name'].isin(['Alaska', 'Hawaii', 'Puerto Rico'])]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot the USA map as background
    usa.plot(ax=ax, color='lightgray', edgecolor='black', zorder=1, alpha=0.7)

    # Use hexbin for point density visualization
    hb = ax.hexbin(df['longitude'], df['latitude'], gridsize=gridsize, 
                    cmap='viridis', mincnt=1, alpha=0.8, zorder=2)
    plt.colorbar(hb, ax=ax, label='Number of Images')

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

    print("\nDensity Map Completed")

def plot_s2_cell_maps(configs, trained_partitions, df_train, world_map):
    """
    Visualizes the resulting S2 cell partitions on a map for comparison.
    Each configuration (with different tau_max) is plotted side-by-side.
    """
    fig, axes = plt.subplots(1, len(configs), figsize=(20, 10))
    # Handle single configuration case where axes is not a list
    if len(configs) == 1: 
        axes = [axes]

    for i, cfg in enumerate(configs):
        ax = axes[i]
        # Plot background map
        world_map.plot(ax=ax, color='lightgrey')
        
        # Get the list of leaf cells for this specific configuration
        leaves = trained_partitions[cfg['name']]
        
        # Iterate through leaf cells and draw their boundaries
        for leaf in leaves:
            vertices = get_cell_vertices(leaf['cell_id'])
            # Create a polygon patch from S2 cell vertices
            polygon = patches.Polygon(vertices, linewidth=0.5, edgecolor='red', facecolor='none')
            ax.add_patch(polygon)
            
        ax.set_title(f"Partitioning: {cfg['name']} (tau={cfg['tau_max']})")
    
    plt.tight_layout()
    plt.show()