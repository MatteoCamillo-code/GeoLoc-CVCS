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

def plot_s2_cell_maps_trio(configs, trained_partitions, df_train, world_map, save_images=False, output_dir=None):
    """
    Visualizes the resulting S2 cell partitions side-by-side on a map for comparison.
    Optional: Saves the combined plot as a PNG.
    """
    fig, axes = plt.subplots(1, len(configs), figsize=(20, 10))
    # Handle single configuration case where axes is not a list
    if len(configs) == 1: 
        axes = [axes]

    for i, cfg in enumerate(configs):
        ax = axes[i]
        # Plot background map
        world_map.plot(ax=ax, color='lightgrey')
        
        # Get the leaf cells for this specific configuration
        leaves = trained_partitions[cfg['name']]
        
        # Iterate through leaf cells and draw their boundaries
        for leaf in leaves:
            vertices = get_cell_vertices(leaf['cell_id'])
            # Create a polygon patch from S2 cell vertices
            polygon = patches.Polygon(vertices, linewidth=0.5, edgecolor='red', facecolor='none')
            ax.add_patch(polygon)
            
        ax.set_title(f"Partitioning: {cfg['name']} (tau={cfg['tau_max']})")
    
    plt.tight_layout()

    # Saving logic
    if save_images:
        if output_dir is None:
            # Default directory if none provided
            output_dir = "tbd"
        
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "s2_configurations_comparison.png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")

    plt.show()

def plot_s2_cell_maps(configs: list, trained_partitions: dict, df_train: pd.DataFrame, usa: gpd.GeoDataFrame, save_images: bool = False):
    """
    Generates and displays maps visualizing the S2 cells created for each partitioning configuration.
    """
    print("Generating S2 cell maps...")

    fig, axes = plt.subplots(len(configs), 1, figsize=(15, 8 * len(configs)))
    if len(configs) == 1: 
        axes = [axes]

    # Define output directory for saving images
    output_plot_dir = "/content/drive/MyDrive/osv5m/plots"
    if save_images:
        os.makedirs(output_plot_dir, exist_ok=True)

    for i, cfg in enumerate(configs):
        ax = axes[i]
        name = cfg['name']
        leaf_cells = trained_partitions[name]

        # Background USA
        usa.plot(ax=ax, color='#f2f2f2', edgecolor='#999999', zorder=1)

        # Training points
        ax.scatter(df_train['longitude'], df_train['latitude'], s=0.5, color='blue', alpha=0.1, zorder=2)

        # S2 Boundaries
        for leaf in leaf_cells:
            verts = get_cell_vertices(leaf['cell_id'])
            polygon = patches.Polygon(verts, linewidth=0.6, edgecolor='red', facecolor='none', alpha=0.8, zorder=3)
            ax.add_patch(polygon)

        ax.set_title(f"{name} | tau_max={cfg['tau_max']} | Classes: {len(leaf_cells)}")
        ax.set_xlim([-126, -66])
        ax.set_ylim([24, 50])

    plt.tight_layout()
    plt.show()

    if save_images:
        for i, cfg in enumerate(configs):
            name = cfg['name']
            single_fig, single_ax = plt.subplots(1, 1, figsize=(15, 8))
            usa.plot(ax=single_ax, color='#f2f2f2', edgecolor='#999999', zorder=1)
            single_ax.scatter(df_train['longitude'], df_train['latitude'], s=0.5, color='blue', alpha=0.1, zorder=2)
            for leaf in trained_partitions[name]:
                verts = get_cell_vertices(leaf['cell_id'])
                polygon = patches.Polygon(verts, linewidth=0.6, edgecolor='red', facecolor='none', alpha=0.8, zorder=3)
                single_ax.add_patch(polygon)
            single_ax.set_title(f"{name} | tau_max={cfg['tau_max']} | Classes: {len(trained_partitions[name])}")
            single_ax.set_xlim([-126, -66])
            single_ax.set_ylim([24, 50])
            single_fig.tight_layout()
            filename = os.path.join(output_plot_dir, f"s2_cell_map_{name}.png")
            single_fig.savefig(filename, dpi=300)
            plt.close(single_fig) 
        print(f"All S2 cell maps saved to {output_plot_dir}/")

    print("\nS2 Cell Map Generation Completed")